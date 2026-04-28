"""
Train a diffusion model on images.
"""
import os
import wandb
from mpi4py import MPI

os.environ["WANDB_API_KEY"] = '389f9942e9b10a034547c47a26e9c987effb0c42'
os.environ["WANDB_MODE"] = "offline"

import sys
sys.path.append('/data/yjy_data/DDBM')
import argparse
from ddbm import dist_util, logger
from datasets import load_data
from ddbm.resample import create_named_schedule_sampler
from ddbm.script_util_2 import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    sample_defaults,
    args_to_dict,
    add_dict_to_argparser,
    get_workdir
)
from ddbm.train_util_2 import TrainLoop
from SAB.ConvNetworkWithImageFeature_2 import ConvNetworkWithImageFeature as CNW
import torch.distributed as dist

from pathlib import Path

from glob import glob
import os
from datasets.augment import AugmentPipe


def main(args):
    # 先告诉 dist_util 我们要用哪块 GPU
    dist_util.set_device_id(args.gpu)
    workdir = get_workdir(args.exp)
    Path(workdir).mkdir(parents=True, exist_ok=True)

    dist_util.setup_dist()
    logger.configure(dir=workdir)
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # visible_gpus = [3, 4]  # 指定 GPU 列表
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_gpus[rank % len(visible_gpus)])
    # # 正常的训练逻辑
    # print(f"Process {rank} is using GPU {os.environ['CUDA_VISIBLE_DEVICES']}")
    if dist.get_rank() == 0:
        name = args.exp if args.resume_checkpoint1 == "" else args.exp + '_resume'
        wandb.init(project="S2O_canny_scene", group=args.exp, name=name, config=vars(args),
                   mode='online' if not args.debug else 'disabled')
        logger.log("creating model and diffusion...")

    data_image_size = args.image_size

    if args.resume_checkpoint == "":
        model_ckpts = list(glob(f'{workdir}/*model*[0-9].*'))
        # 不用SAB
        # if len(model_ckpts) > 0:
        #     max_ckpt = max(model_ckpts, key=lambda x: int(x.split('model_')[-1].split('.')[0]))
        #     if os.path.exists(max_ckpt):
        #         args.resume_checkpoint = max_ckpt
        #         if dist.get_rank() == 0:
        #             logger.log('Resuming from checkpoint: ', max_ckpt)
        # 从这里开始**********是用了SAB*************
        if len(model_ckpts) > 0:
            # Sort checkpoints by the model number and the numeric part after the model identifier
            model_ckpts.sort(key=lambda x: (
                int(x.split('model_')[1].split('_')[0]),  # Extract the model identifier (1, 2, etc.)
                int(x.split('_')[-1].split('.')[0])  # Extract the numeric suffix (like 010000)
            ), reverse=True)  # Sort in descending order by model identifier and numeric suffix

            # Dictionary to store the most recent checkpoint for each model identifier
            latest_ckpts = {}

            for checkpoint in model_ckpts:
                model_id = int(checkpoint.split('model_')[1].split('_')[0])  # Extract model number (e.g., 1, 2)
                if model_id not in latest_ckpts:
                    latest_ckpts[model_id] = checkpoint  # Store the first checkpoint found for the model
                else:
                    # Compare the numeric suffix and select the one with the largest suffix
                    current_suffix = int(checkpoint.split('_')[-1].split('.')[0])  # Extract numeric suffix
                    stored_suffix = int(latest_ckpts[model_id].split('_')[-1].split('.')[0])
                    if current_suffix > stored_suffix:
                        latest_ckpts[model_id] = checkpoint  # Update to the more recent checkpoint

            # Assign the top two checkpoints to resume_checkpoint1 and resume_checkpoint2
            resume_checkpoint_1 = latest_ckpts.get(1)  # Get the checkpoint for model_1
            resume_checkpoint_2 = latest_ckpts.get(2)  # Get the checkpoint for model_2

            # If the checkpoints are found, assign them to the args variables
            if resume_checkpoint_1 and os.path.exists(resume_checkpoint_1):
                args.resume_checkpoint1 = resume_checkpoint_1
                if dist.get_rank() == 0:
                    logger.log(f'Resuming from checkpoint 1: {resume_checkpoint_1}')

            if resume_checkpoint_2 and os.path.exists(resume_checkpoint_2):
                args.resume_checkpoint2 = resume_checkpoint_2
                if dist.get_rank() == 0:
                    logger.log(f'Resuming from checkpoint 2: {resume_checkpoint_2}')
        # ****************到这******************

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    # 加算子

    CNW_net = CNW(3, 3)
    CNW_net.to(dist_util.dev())
    ##
    if dist.get_rank() == 0:
        wandb.watch(model, log='all')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size() * batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

    if dist.get_rank() == 0:
        logger.log("creating data loader...")

    data, test_data = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=batch_size,
        image_size=data_image_size,
        num_workers=args.num_workers,
    )

    if args.use_augment:
        augment = AugmentPipe(
            p=0.12, xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1
        )
    else:
        augment = None

    logger.log("training...")
    TrainLoop(
        model=model,
        CNW_net = CNW_net,
        diffusion=diffusion,
        train_data=data,
        test_data=test_data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        test_interval=args.test_interval,
        save_interval=args.save_interval,
        save_interval_for_preemption=args.save_interval_for_preemption,
        resume_checkpoint=args.resume_checkpoint,
        resume_checkpoint1=args.resume_checkpoint1,
        resume_checkpoint2=args.resume_checkpoint2,
        workdir=workdir,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        augment_pipe=augment,
        **sample_defaults()
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="/data/hjf/Dataset/SEN12_Scene/combine",
        dataset='SEN12',
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=10,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        test_interval=500,
        save_interval=10000,
        save_interval_for_preemption=50000,
        resume_checkpoint="",
        resume_checkpoint1="",
        resume_checkpoint2="",
        exp='logs_S2O_Canny_CAIB_MSFM_scene_noE2',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        debug=False,
        num_workers=2,
        use_augment=False,
        gpu=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
