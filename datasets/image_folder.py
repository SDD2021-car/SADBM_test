import torch
import torch.utils.data as data

from PIL import Image
import os
import os.path
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

from datasets import DistInfiniteBatchSampler

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


def load_data(
        data_dir,
        dataset,
        batch_size,
        image_size,
        deterministic=False,
        include_test=False,
        seed=42,
        num_workers=2,
):
    # Compute batch size for this worker.
    root = data_dir

    if dataset == 'edges2handbags':

        from .aligned_dataset import EdgesDataset
        trainset = EdgesDataset(dataroot=root, train=True, img_size=image_size,
                                random_crop=True, random_flip=True)

        valset = EdgesDataset(dataroot=root, train=True, img_size=image_size,
                              random_crop=False, random_flip=False)
        if include_test:
            testset = EdgesDataset(dataroot=root, train=False, img_size=image_size,
                                   random_crop=False, random_flip=False)

    elif dataset == 'diode':

        from .aligned_dataset import DIODE
        trainset = DIODE(dataroot=root, train=True, img_size=image_size,
                         random_crop=True, random_flip=True, disable_cache=True)

        valset = DIODE(dataroot=root, train=True, img_size=image_size,
                       random_crop=False, random_flip=False, disable_cache=True)

        if include_test:
            testset = DIODE(dataroot=root, train=False, img_size=image_size,
                            random_crop=False, random_flip=False)

    loader = DataLoader(
        dataset=trainset, num_workers=num_workers, pin_memory=True,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(trainset), glb_batch_size=batch_size * dist.get_world_size(), seed=seed,
            shuffle=not deterministic, filling=True, rank=dist.get_rank(), world_size=dist.get_world_size(),
        )
    )

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler = torch.utils.data.DistributedSampler(
        valset, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers, drop_last=False)

    if include_test:

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler = torch.utils.data.DistributedSampler(
            testset, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=False
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size,
            sampler=sampler, num_workers=num_workers, shuffle=False, drop_last=False)

        return loader, val_loader, test_loader
    else:
        return loader, val_loader