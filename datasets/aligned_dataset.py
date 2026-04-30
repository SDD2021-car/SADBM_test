import os.path
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from datasets.image_folder import make_dataset
from PIL import Image
from torchvision.utils import save_image
import torchvision
import blobfile as bf
from secrets import randbelow
from glob import glob
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
from torchvision.utils import save_image
import json

from PIL import Image
import matplotlib.pyplot as plt

def get_params(size, resize_size, crop_size):
    w, h = size
    new_h = h
    new_w = w

    ss, ls = min(w, h), max(w, h)  # shortside and longside
    width_is_shorter = w == ss
    ls = int(resize_size * ls / ss)
    ss = resize_size
    new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(params, resize_size, crop_size, method=Image.BICUBIC, flip=True, crop=True, totensor=True):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: __scale(img, resize_size, method)))
    # transform_list.append(transforms.Lambda(lambda img: __random_crop(img, crop_size)))

    if flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    if totensor:
        transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def get_transform2(params, resize_size, crop_size, method=Image.BICUBIC, flip=True, crop=True, totensor=True):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: __scale(img, resize_size, method)))
    # transform_list.append(transforms.CenterCrop(crop_size))

    if flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    if totensor:
        transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __random_crop(img, crop_size):
    w, h = img.size
    th, tw = crop_size, crop_size
    if w == tw and h == th:
        return img

    x1 = randbelow(w - tw + 1)
    y1 = randbelow(h - th + 1)
    return img.crop((x1, y1, x1 + tw, y1 + th))

def __scale(img, target_width, method=Image.BICUBIC):
    if isinstance(img, torch.Tensor):
        # 加随机裁剪
        return torch.nn.functional.interpolate(img.unsqueeze(0), size=(target_width, target_width), mode='bicubic',
                                               align_corners=False).squeeze(0)
    else:
        return img.resize((target_width, target_width), method)


def __flip(img, flip):
    if flip:
        if isinstance(img, torch.Tensor):
            return img.flip(-1)
        else:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def get_flip(img, flip):
    return __flip(img, flip)




class EdgesDataset(torch.utils.data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dataroot, train=True, img_size=256, random_crop=False, random_flip=True, answer_json_path=""):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        self.dataroot = dataroot
        if train:
            self.train_dir = os.path.join(dataroot, 'train')  # get the image directory
            self.train_paths = make_dataset(self.train_dir)  # get image paths
            self.AB_paths = sorted(self.train_paths)
        else:

            self.test_dir = os.path.join(dataroot, 'val')  # get the image directory

            self.AB_paths = make_dataset(self.test_dir)  # get image paths

        self.crop_size = img_size
        self.resize_size = img_size

        self.random_crop = random_crop
        self.random_flip = random_flip
        self.train = train
        self.answer_json_path = answer_json_path
        self._build_answer_map()



    def _load_answer_from_json_file(self, json_path):
        """Support two formats:
        1) sidecar dict: {"answer": "..."}
        2) list manifest: [{"image": "...", "answer": "..."}, ...]
        """
        if not os.path.exists(json_path):
            return None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        if isinstance(data, dict):
            if "answer" in data:
                return str(data.get("answer", ""))
            return None

        if isinstance(data, list):
            # list manifest does not correspond to one sample directly
            return None

        return None

    def _tail_key(self, path_str):
        p = os.path.normpath(path_str)
        parts = p.split(os.sep)
        for anchor in [("Train", "B"), ("train",), ("Val", "B"), ("val",)]:
            n = len(anchor)
            for i in range(len(parts) - n):
                if tuple(parts[i:i+n]) == anchor:
                    return os.path.join(*parts[i+n:]) if i+n < len(parts) else ""
        return os.path.basename(p)

    def _build_answer_map(self):
        """Build mapping: basename(image_path) -> answer for manifest json files under dataroot."""
        self.answer_map = {}
        self.answer_map_stem = {}
        self.answer_map_full = {}
        self.answer_map_tail = {}
        scan_roots = [self.dataroot]
        if self.answer_json_path:
            scan_roots = [self.answer_json_path] if os.path.isfile(self.answer_json_path) else [self.answer_json_path]

        for scan_root in scan_roots:
            if os.path.isfile(scan_root):
                json_candidates = [scan_root]
            else:
                json_candidates = []
                for root, _, files in os.walk(scan_root):
                    for fn in files:
                        if fn.lower().endswith('.json'):
                            json_candidates.append(os.path.join(root, fn))

            for json_path in json_candidates:
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    continue

                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        image_path = item.get('image', '')
                        answer = item.get('answer', '')
                        if image_path:
                            ans = str(answer)
                            norm_full = os.path.normpath(image_path)
                            base = os.path.basename(norm_full)
                            stem = os.path.splitext(base)[0]
                            self.answer_map_full[norm_full] = ans
                            self.answer_map[base] = ans
                            self.answer_map_stem[stem] = ans
                            tail = self._tail_key(norm_full)
                            if tail:
                                self.answer_map_tail[tail] = ans

    def _get_answer(self, ab_path):
        # Priority 1: sidecar same-name json
        sidecar_json = os.path.splitext(ab_path)[0] + '.json'
        sidecar_answer = self._load_answer_from_json_file(sidecar_json)
        if sidecar_answer is not None:
            return sidecar_answer

        norm = os.path.normpath(ab_path)
        base = os.path.basename(norm)
        stem = os.path.splitext(base)[0]

        # Priority 2: manifest exact full-path key
        if norm in self.answer_map_full:
            return self.answer_map_full[norm]

        # Priority 3: train/val tail key
        tail = self._tail_key(norm)
        if tail in self.answer_map_tail:
            return self.answer_map_tail[tail]

        # Priority 4: basename key
        if base in self.answer_map:
            return self.answer_map[base]

        # Priority 5: stem key
        return self.answer_map_stem.get(stem, "")

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        params1 = get_params(A.size, self.resize_size, self.crop_size)
        # params2 = get_params(A.size, self.resize_size, self.crop_size)
        transform_image = get_transform(params1, self.resize_size, self.crop_size, crop=self.random_crop,
                                        flip=self.random_flip)
        # transform_imageB = get_transform(params1, self.resize_size, self.crop_size, crop=self.random_crop,
        #                                 flip=self.random_flip)
        A = transform_image(A)
        B = transform_image(B)
        answer = self._get_answer(AB_path)
        if not self.train:
            # return B, A, index, AB_path
            return B, A, index, AB_path, answer
        else:
            # return B, A, index
            return B, A, index, answer

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)



class DIODE(torch.utils.data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dataroot, train=True,  img_size=256, random_crop=False, random_flip=True, down_sample_img_size = 0, cache_name='cache', disable_cache=False):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        self.dataroot = dataroot
        self.image_root = os.path.join(dataroot, 'train' if train else 'val')
        self.crop_size = img_size
        self.resize_size = img_size
        
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.train = train
        self.filenames = [l for l in os.listdir(self.image_root) if not l.endswith('.pth') and not l.endswith('_depth.png') and not l.endswith('_normal.png')]

        self.cache_path = os.path.join(self.image_root, cache_name+f'_{img_size}.pth')
        if os.path.exists(self.cache_path) and not disable_cache:
            self.cache = torch.load(self.cache_path)
            # self.cache['img'] = self.cache['img'][:256]
            self.scale_factor = self.cache['scale_factor']
            print('Loaded cache from {}'.format(self.cache_path))
        else:
            self.cache = None

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        
        fn = self.filenames[index]
        img_path = os.path.join(self.image_root, fn)
        label_path = os.path.join(self.image_root, fn[:-4]+'_normal.png')

        with bf.BlobFile(img_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        with bf.BlobFile(label_path, "rb") as f:
            pil_label = Image.open(f)
            pil_label.load()
        pil_label = pil_label.convert("RGB")

        # apply the same transform to both A and B
        params =  get_params(pil_image.size, self.resize_size, self.crop_size)

        transform_label = get_transform(params, self.resize_size, self.crop_size, method=Image.NEAREST, crop =False, flip=self.random_flip)
        transform_image = get_transform( params, self.resize_size, self.crop_size, crop =False, flip=self.random_flip)

        cond = transform_label(pil_label)
        img = transform_image(pil_image)

        # if self.down_sample_img:
        #     image_pil = np.array(image_pil).astype(np.uint8)
        #     down_sampled_image = self.down_sample_img(image=image_pil)["image"]
        #     down_sampled_image = get_tensor()(down_sampled_image)
        #     # down_sampled_image = transforms.ColorJitter(brightness = [0.85,1.15], contrast=[0.95,1.05], saturation=[0.95,1.05])(down_sampled_image)
        #     data_dict = {"ref":label_tensor, "low_res":down_sampled_image, "ref_ori":label_tensor_ori, "path": path}

        #     return image_tensor, data_dict
        if not self.train:
            return img, cond, index, fn
        else:
            return img, cond, index
        
    

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.cache is not None:
            return len(self.cache['img'])
        else:
            return len(self.filenames)
    

