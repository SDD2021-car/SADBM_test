"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torchvision.utils as vutils
import torch.distributed as dist
from torchvision.utils import make_grid
from torchvision import transforms as T
from ddbm import dist_util, logger
from ddbm.script_util_2 import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from ddbm.random_util import get_generator
from ddbm.karras_diffusion import karras_sample, forward_sample
from datasets.image_folder import load_data
from SAB.ConvNetworkWithImageFeature import ConvNetworkWithImageFeature as CNW
from pathlib import Path
import Sobel_train

from PIL import Image
def get_workdir(exp):
    workdir = f'{exp}'
    return workdir

def tensor2image1(x: th.Tensor, i: int):
    x = ((x + 1) * 127.5).clamp(0, 255).to(th.uint8)
    imgs = x.detach().to('cpu')

    img = make_grid(imgs)
    img = T.functional.to_pil_image(img)
    fold_path = '/data/yjy_data/DDBM/result_SAR2OPT_256'
    file_name = f'GT-{i}.jpg'
    os.makedirs(fold_path,exist_ok=True)
    full_file_path = os.path.join(fold_path, file_name)
    img.save(full_file_path)
    return img

def tensor2image(x: th.Tensor, i: int):
    x = ((x + 1) * 127.5).clamp(0, 255).to(th.uint8)
    imgs = x.detach().to('cpu')

    img = make_grid(imgs)
    img = T.functional.to_pil_image(img)
    fold_path = '/data/yjy_data/DDBM/result_SAR2OPT_256'
    file_name = f'sample-{i}.jpg'
    os.makedirs(fold_path,exist_ok=True)
    full_file_path = os.path.join(fold_path, file_name)
    img.save(full_file_path)
    return img


def save_images_with_filenames(images, file_names, output_dir="output_images"):
    """
    将一个张量 (N, C, H, W) 中的每个图像保存为单独的文件，使用指定的文件名。

    Args:
        images (torch.Tensor): 图像张量，形状为 (N, C, H, W)，值可以在任意范围。
        file_names (list): 包含 N 个文件名的列表。
        output_dir (str): 输出目录，默认是 "output_images"。

    Returns:
        None
    """
    # if not isinstance(images, th.Tensor):
    #     raise ValueError("`images` 必须是一个 PyTorch 张量。")
    # if len(images) != len(file_names):
    #     raise ValueError("`images` 的数量和 `file_names` 的数量必须一致。")
    # 创建保存目录
    os.makedirs(output_dir, exist_ok=True)
    # 定义转换函数
    to_pil_image = T.ToPILImage()
    # 遍历保存每张图片
    for i, file_name in enumerate(file_names):
        file_name = os.path.basename(file_name)
        image_tensor = images[i]  # 取出单张图片

        image_tensor = ((image_tensor + 1) * 127.5).clamp(0, 255).to(th.uint8)
        imgs = image_tensor.detach().to('cpu')
        # 转为 PIL 图像
        image = to_pil_image(imgs)
        # 保存为文件
        image.save(os.path.join(output_dir,file_name))

def main():
    args = create_argparser().parse_args()

    workdir = os.path.dirname(args.model_path)

    ## assume ema ckpt format: ema_{rate}_{steps}.pt
    split = args.model_path.split("_")
    step = int(split[-1].split(".")[0])
    sample_dir = Path(workdir)/f'sample_{step}/w={args.guidance}_churn={args.churn_step_ratio}'
    dist_util.setup_dist()
    if dist.get_rank() == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=workdir)


    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model = model.to(dist_util.dev())

    # canny_refine_network = Canny_train.UNet(in_channels=1, out_channels=1)
    # checkpoint = th.load(args.path_refine_network, map_location='cpu')  # 或者 'cuda:7' 根据你的设备
    # # 支持两种情况：保存的是整个模型或 state_dict
    # if 'state_dict' in checkpoint:
    #     canny_refine_network.load_state_dict(checkpoint['state_dict'])
    # else:
    #     canny_refine_network.load_state_dict(checkpoint)
    #
    # # 冻结参数（不参与反向传播）
    # for param in canny_refine_network.parameters():
    #     param.requires_grad = False
    # canny_refine_network = canny_refine_network.to(dist_util.dev())
    # # 切换到 eval 模式（关闭 Dropout / BatchNorm 统计更新）
    # canny_refine_network.eval()

    CNW_net = CNW(3, 3, True)
    CNW_net.load_state_dict(
        dist_util.load_state_dict(args.CNW_path, map_location="cpu")
    )
    CNW_net = CNW_net.to(dist_util.dev())

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    

    all_images = []
    

    all_dataloaders = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        include_test=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    if args.split =='train':
        dataloader = all_dataloaders[1]
    elif args.split == 'test':
        dataloader = all_dataloaders[2]
    else:
        raise NotImplementedError
    args.num_samples = len(dataloader.dataset)


    for i, data in enumerate(dataloader):
        
        x0_image = data[0]
        x0_filename = data[3]
        x0 = x0_image.to(dist_util.dev()) * 2 -1
        
        y0_image = data[1].to(dist_util.dev())

        alpha = [0.01, 0.01, 0.01, 0.01]
        cond = CNW_net(y0_image, alpha)

        y0 = cond.to(dist_util.dev()) * 2 - 1
        model_kwargs = {'xT': y0}
        index = data[2].to(dist_util.dev())

        sample, path, nfe = karras_sample(
            diffusion,
            model,
            y0,
            x0,
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=diffusion.sigma_min,
            sigma_max=diffusion.sigma_max,
            churn_step_ratio=args.churn_step_ratio,
            rho=args.rho,
            guidance=args.guidance
        )
        # for i, x in enumerate([x0]):
        #     tensor2image1(x, i)
        # for i, x in enumerate(path):
        #     tensor2image(x, i)
        save_images_with_filenames(path[200],x0_filename,'/data/yjy_data/DDBM_GT_Unet/result_S2O_Canny_130000')

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        if index is not None:
            gathered_index = [th.zeros_like(index) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_index, index)
            gathered_samples = th.cat(gathered_samples)
            gathered_index = th.cat(gathered_index)
            gathered_samples = gathered_samples[th.argsort(gathered_index)]
        else:
            gathered_samples = th.cat(gathered_samples)

        num_display = min(64, sample.shape[0])
        if i == 0 and dist.get_rank() == 0:
            vutils.save_image(sample.permute(0,3,1,2)[:num_display].float(), f'{sample_dir}/sample_{i}.png', normalize=True,  nrow=int(np.sqrt(num_display)))
            if x0 is not None:
                vutils.save_image(x0_image[:num_display], f'{sample_dir}/x_{i}.png',nrow=int(np.sqrt(num_display)))
            vutils.save_image(y0_image[:num_display]/2+0.5, f'{sample_dir}/y_{i}.png',nrow=int(np.sqrt(num_display)))
            
            
        all_images.append(gathered_samples.detach().cpu().numpy())
        
        
    logger.log(f"created {len(all_images) * args.batch_size * dist.get_world_size()} samples")
        

    arr = np.concatenate(all_images, axis=0)
    arr = arr[:args.num_samples]
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(sample_dir, f"samples_{shape_str}_nfe{nfe}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="/data/yjy_data/dataset/sen_data_new/combine", ## only used in bridge
        dataset='edges2handbags',
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        split='test',
        churn_step_ratio=0.33,
        rho=7.0,
        steps=100,
        model_path="/data/yjy_data/DDBM_GT_Unet/scripts/logs_S2O_Canny_CAIB_MSFM/ema_2_0.9999_130000.pt",
        CNW_path="/data/yjy_data/DDBM_GT_Unet/scripts/logs_S2O_Canny_CAIB_MSFM/ema_1_0.9999_130000.pt",
        path_refine_network="/data/yjy_data/DDBM_GT_Unet/canny_optimization_result/train_result/unet_epoch_50.pth",
        exp="logs_S2O_Canny_CAIB_MSFM",
        seed=42,
        ts="",
        upscale=False,
        num_workers=2,
        guidance=0.5,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
