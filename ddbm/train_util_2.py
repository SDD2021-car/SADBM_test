import copy
import functools
import os
from pathlib import Path
from collections import OrderedDict
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from SAB.ConvNetworkWithImageFeature import ConvNetworkWithImageFeature as CNW
from ddbm.random_util import get_generator
from .fp16_util import (
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)
import numpy as np
from ddbm.script_util_2 import NUM_CLASSES

from ddbm.karras_diffusion import karras_sample

from ddbm.text_condition import TextConditionEncoder

import glob 
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.

INITIAL_LOG_LOSS_SCALE = 20.0

import wandb

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        CNW_net,
        diffusion,
        train_data,
        test_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        test_interval,
        save_interval,
        save_interval_for_preemption,
        resume_checkpoint1,
        resume_checkpoint2,
        workdir,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        total_training_steps=10000000,
        augment_pipe=None,
        text_model_path="",
        text_guidance_weight=1.0,
        text_cache_size=4096,
        **sample_kwargs,
    ):
        self.model = model
        #
        self.CNW_net = CNW_net
        #
        self.diffusion = diffusion
        self.data = train_data
        self.test_data = test_data
        self.image_size = model.image_size
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.workdir = workdir
        self.test_interval = test_interval
        self.save_interval = save_interval
        self.save_interval_for_preemption = save_interval_for_preemption
        self.resume_checkpoint1 = resume_checkpoint1
        self.resume_checkpoint2 = resume_checkpoint2
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.total_training_steps = total_training_steps
        self.text_guidance_weight = text_guidance_weight
        self.text_cache_size = max(int(text_cache_size), 0)
        self._text_feat_cache = OrderedDict()
        self.step = 0
        self.resume_step = 0
        self.resume_step1 = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()


        self._load_and_sync_parameters()
        self.mp_trainer2 = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        #
        self.mp_trainer1 = MixedPrecisionTrainer(
            model=self.CNW_net,
            use_fp16=False,
            fp16_scale_growth=fp16_scale_growth,
        )
        #
        self.opt1 = RAdam(
            self.mp_trainer1.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        self.opt2 = RAdam(
            self.mp_trainer2.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params1,self.ema_params2 = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params1 = [
                copy.deepcopy(self.mp_trainer1.master_params)
                for _ in range(len(self.ema_rate))
            ]
            self.ema_params2 = [
                copy.deepcopy(self.mp_trainer2.master_params)
                for _ in range(len(self.ema_rate))
            ]
        use_dataparallel = os.environ.get("DDBM_USE_DATAPARALLEL", "0") == "1"
        if th.cuda.is_available() and use_dataparallel and th.cuda.device_count() > 1:
            self.use_ddp = False
            self.ddp_model = th.nn.DataParallel(self.model)
            logger.log(f"Using DataParallel on {th.cuda.device_count()} GPUs (single-process mode).")
        elif th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self.step = self.resume_step1

        self.generator = get_generator(sample_kwargs['generator'], self.batch_size,42)
        self.sample_kwargs = sample_kwargs

        self.augment = augment_pipe

        self.stage0_train = False
        self.fusion_canny = True
        self.canny_SAR = None
        self.text_encoder = TextConditionEncoder(model_name=(text_model_path or "openai/clip-vit-base-patch32"), device=dist_util.dev(), local_files_only=True) if getattr(self.model, "use_text_guidance", False) else None
    

    def _encode_text_with_cache(self, answers):
        if self.text_encoder is None:
            return None
        answer_list = list(answers)
        if self.text_cache_size <= 0:
            return self.text_encoder.encode(answer_list)

        uncached = [a for a in answer_list if a not in self._text_feat_cache]
        if uncached:
            encoded_new = self.text_encoder.encode(uncached)
            for ans, feat in zip(uncached, encoded_new):
                self._text_feat_cache[ans] = feat.detach().cpu()
                self._text_feat_cache.move_to_end(ans)
                if len(self._text_feat_cache) > self.text_cache_size:
                    self._text_feat_cache.popitem(last=False)

        features = [self._text_feat_cache[a].to(dist_util.dev()) for a in answer_list]
        return th.stack(features, dim=0)


    def _load_and_sync_parameters(self):
        resume_checkpoint1 = find_resume_checkpoint() or self.resume_checkpoint1
        resume_checkpoint2 = find_resume_checkpoint() or self.resume_checkpoint2
        if resume_checkpoint1:
            self.resume_step1 = parse_resume_step_from_filename(resume_checkpoint1)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint1}...")
                logger.log('Resume step: ', self.resume_step1)

            # self.CNW_net.load_state_dict(
            #     # dist_util.load_state_dict(
            #     #     resume_checkpoint, map_location=dist_util.dev()
            #     # ),
            #     th.load(resume_checkpoint1, map_location=dist_util.dev()),
            # )
            state_dict1 = th.load(resume_checkpoint1, map_location=dist_util.dev())
            incompatible1 = self.CNW_net.load_state_dict(state_dict1, strict=False)
            if dist.get_rank() == 0:
                logger.log(f"CNW missing keys ({len(incompatible1.missing_keys)}): {incompatible1.missing_keys}")
                logger.log(f"CNW unexpected keys ({len(incompatible1.unexpected_keys)}): {incompatible1.unexpected_keys}")

            dist.barrier()
        if resume_checkpoint2:
            self.resume_step2 = parse_resume_step_from_filename(resume_checkpoint2)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint2}...")
                logger.log('Resume step: ', self.resume_step2)

            # self.model.load_state_dict(
            #     # dist_util.load_state_dict(
            #     #     resume_checkpoint, map_location=dist_util.dev()
            #     # ),
            #     th.load(resume_checkpoint2, map_location=dist_util.dev()),
            # )
            state_dict2 = th.load(resume_checkpoint2, map_location=dist_util.dev())
            incompatible2 = self.model.load_state_dict(state_dict2, strict=False)
            if dist.get_rank() == 0:
                logger.log(f"Model missing keys ({len(incompatible2.missing_keys)}): {incompatible2.missing_keys}")
                logger.log(f"Model unexpected keys ({len(incompatible2.unexpected_keys)}): {incompatible2.unexpected_keys}")

            dist.barrier()

    def _load_ema_parameters(self, rate):
        ema_params1 = copy.deepcopy(self.mp_trainer1.master_params)
        ema_params2 = copy.deepcopy(self.mp_trainer2.master_params)
        main_checkpoint1 = find_resume_checkpoint() or self.resume_checkpoint1
        main_checkpoint2 = find_resume_checkpoint() or self.resume_checkpoint2
        ema_checkpoint1 = find_ema_checkpoint(main_checkpoint1, self.resume_step, rate)
        ema_checkpoint2 = find_ema_checkpoint(main_checkpoint2, self.resume_step, rate)
        if ema_checkpoint1:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint1}...")

            state_dict = th.load(ema_checkpoint1, map_location=dist_util.dev())
            ema_params2 = self.mp_trainer2.state_dict_to_master_params(state_dict)
            dist.barrier()

        if ema_checkpoint2:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint2}...")

            state_dict = th.load(ema_checkpoint2, map_location=dist_util.dev())
            ema_params1 = self.mp_trainer1.state_dict_to_master_params(state_dict)
            dist.barrier()
        return ema_params1,ema_params2

    # def _load_optimizer_state(self):
    #     main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
    #     if main_checkpoint.split('/')[-1].startswith("latest"):
    #         prefix = 'latest_'
    #     else:
    #         prefix = ''
    #     opt_checkpoint = bf.join(
    #         bf.dirname(main_checkpoint), f"{prefix}opt{self.resume_step:06}.pt"
    #     )
    #     if bf.exists(opt_checkpoint):
    #         logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
    #         # state_dict = dist_util.load_state_dict(
    #         #     opt_checkpoint, map_location=dist_util.dev()
    #         # )
    #         state_dict = th.load(opt_checkpoint, map_location=dist_util.dev())
    #         self.opt1.load_state_dict(state_dict)
    #         dist.barrier()
    def _load_optimizer_state(self):
        # Select main checkpoints for model 1 and model 2
        main_checkpoint1 = find_resume_checkpoint() or self.resume_checkpoint1
        main_checkpoint2 = find_resume_checkpoint() or self.resume_checkpoint2

        # Determine prefix based on the checkpoint name
        if main_checkpoint1.split('/')[-1].startswith("latest"):
            prefix1 = 'latest_'
        else:
            prefix1 = ''

        if main_checkpoint2.split('/')[-1].startswith("latest"):
            prefix2 = 'latest_'
        else:
            prefix2 = ''

        # Construct the path to the optimizer checkpoint for model 1
        opt_checkpoint1 = bf.join(bf.dirname(main_checkpoint1), f"{prefix1}opt{self.resume_step:06}.pt")
        # Construct the path to the optimizer checkpoint for model 2
        opt_checkpoint2 = bf.join(bf.dirname(main_checkpoint2), f"{prefix2}opt{self.resume_step:06}.pt")

        # Load optimizer state for model 1 if the checkpoint exists
        if bf.exists(opt_checkpoint1):
            logger.log(f"loading optimizer state for model 1 from checkpoint: {opt_checkpoint1}")
            state_dict1 = th.load(opt_checkpoint1, map_location=dist_util.dev())
            self.opt1.load_state_dict(state_dict1)
            dist.barrier()

        # Load optimizer state for model 2 if the checkpoint exists
        if bf.exists(opt_checkpoint2):
            logger.log(f"loading optimizer state for model 2 from checkpoint: {opt_checkpoint2}")
            state_dict2 = th.load(opt_checkpoint2, map_location=dist_util.dev())
            self.opt2.load_state_dict(state_dict2)
            dist.barrier()

    def preprocess(self, x):
        if x.shape[1] == 3:
            x =  x * 2 - 1
        return x

    def run_loop(self):
        while True:
            # for batch, cond, _ in self.data:
            #######3
            for data_item in self.data:
                if len(data_item) >= 4:
                    batch, cond, _ = data_item[:3]
                    answer = data_item[3]
                else:
                    batch, cond, _ = data_item
                    answer = [""] * batch.shape[0]
            ######
                if not (not self.lr_anneal_steps or self.step < self.total_training_steps):
                    # Save the last checkpoint if it wasn't already saved.
                    if (self.step - 1) % self.save_interval != 0:
                        self.save()
                    return
                # scale to [-1, 1]
                # batch is OPT | cond is SAR
                batch = self.preprocess(batch)

                alpha = [0.01, 0.01, 0.01, 0.01]

                cond = cond.to(dist_util.dev())
                if self.stage0_train:
                    cond, canny_SAR = self.CNW_net(cond, alpha)
                    self.canny_SAR = canny_SAR
                else:
                    cond = self.CNW_net(cond, alpha)

                cond = cond.cpu()

                if self.augment is not None:
                    batch, _ = self.augment(batch)
                if isinstance(cond, th.Tensor) and batch.ndim == cond.ndim:
                    xT = self.preprocess(cond)
                    cond = {'xT': xT}
                else:
                    cond['xT'] = self.preprocess(cond['xT'])

                ########
                if self.text_encoder is not None:
                    text_feat = self._encode_text_with_cache(answer)
                    cond['text_feat'] = text_feat * self.text_guidance_weight
                #########

                took_step = self.run_step(batch, cond)
                if took_step and self.step % self.log_interval == 0:
                    logs = logger.dumpkvs()

                    if dist.get_rank() == 0:
                        wandb.log(logs, step=self.step)
                        
                if took_step and self.step % self.save_interval == 0:
                    self.save()
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                    
                    test_data_item = next(iter(self.test_data))
                    test_batch, test_cond, _ = test_data_item[:3]
                    test_answer = test_data_item[3] if len(test_data_item) > 3 else None
                    test_cond = test_cond.to(dist_util.dev())
                    test_cond = self.CNW_net(test_cond, alpha)
                    test_cond = test_cond.cpu()

                    if isinstance(test_cond, th.Tensor) and test_batch.ndim == test_cond.ndim:
                        test_cond = {'xT': self.preprocess(test_cond)}
                    else:
                        test_cond['xT'] = self.preprocess(test_cond['xT'])

                    if self.text_encoder is not None and test_answer is not None:
                        test_text_feat = self._encode_text_with_cache(test_answer)
                        test_cond['text_feat'] = test_text_feat * self.text_guidance_weight
                    self.run_test_step(test_batch, test_cond)
                    logs = logger.dumpkvs()

                    if dist.get_rank() == 0:
                        wandb.log(logs, step=self.step)
                

                if took_step and self.step % self.save_interval_for_preemption == 0:
                    self.save(for_preemption=True)
        

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step1 = self.mp_trainer1.optimize(self.opt1)
        took_step2 = self.mp_trainer2.optimize(self.opt2)
        if took_step1 and took_step2:
            self.step += 1
            self._update_ema()

        self._anneal_lr()
        self.log_step()
        return took_step1,took_step2

    def run_test_step(self, batch, cond):
        with th.no_grad():
            self.forward_backward(batch, cond, train=False)

    def forward_backward(self, batch, cond, train=True):
        if train:
            self.mp_trainer1.zero_grad()
            self.mp_trainer2.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                    self.diffusion.training_bridge_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                    self_param = self
                )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler) and train:
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k if train else 'test_'+k: v * weights for k, v in losses.items()}
            )
            if train:
                self.mp_trainer1.backward(loss)
                self.mp_trainer2.backward(loss)

    def _update_ema(self):
        # Update EMA for both mp_trainer1 and mp_trainer2
        for rate, params1,params2 in zip(self.ema_rate, self.ema_params1,self.ema_params2):
            # Update EMA for mp_trainer1
            update_ema(params1, self.mp_trainer1.master_params, rate=rate)

            # Update EMA for mp_trainer2
            update_ema(params2, self.mp_trainer2.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt1.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)

    def save(self, for_preemption=False):
        def maybe_delete_earliest(filename):
            wc = filename.split(f'{(self.step):06d}')[0]+'*'
            freq_states = list(glob.glob(os.path.join(get_blob_logdir(), wc)))
            if len(freq_states) > 3:
                earliest = min(freq_states, key=lambda x: x.split('_')[-1].split('.')[0])
                os.remove(earliest)
                    

        # if dist.get_rank() == 0 and for_preemption:
        #     maybe_delete_earliest(get_blob_logdir())
        def save_checkpoint(rate, params,model_id):
            state_dict = self.mp_trainer1.master_params_to_state_dict(params) if model_id == 1 else self.mp_trainer2.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate} for model_id={model_id}...")

                if model_id == 1:
                    filename = f"model_1_{(self.step):06d}.pt" if not rate else f"ema_1_{rate}_{(self.step):06d}.pt"
                else:
                    filename = f"model_2_{(self.step):06d}.pt" if not rate else f"ema_2_{rate}_{(self.step):06d}.pt"

                if for_preemption:
                    filename = f"freq_{filename}"
                    maybe_delete_earliest(filename)

                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        # Save each model in its own file
        for rate, params in zip(self.ema_rate, self.ema_params1):
            save_checkpoint(rate, params, model_id=1)
        for rate, params in zip(self.ema_rate, self.ema_params2):
            save_checkpoint(rate, params, model_id=2)

        if dist.get_rank() == 0:
            filename_opt1 = f"opt1_{(self.step):06d}.pt"
            filename_opt2 = f"opt2_{(self.step):06d}.pt"

            if for_preemption:
                filename_opt1 = f"freq_{filename_opt1}"
                filename_opt2 = f"freq_{filename_opt2}"
                maybe_delete_earliest(filename_opt1)
                maybe_delete_earliest(filename_opt2)

            with bf.BlobFile(bf.join(get_blob_logdir(), filename_opt1), "wb") as f:
                th.save(self.opt1.state_dict(), f)

            with bf.BlobFile(bf.join(get_blob_logdir(), filename_opt2), "wb") as f:
                th.save(self.opt2.state_dict(), f)

            # Save models' parameters
        save_checkpoint(0, self.mp_trainer1.master_params, model_id=1)
        save_checkpoint(0, self.mp_trainer2.master_params, model_id=2)
        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/model_1_010000.pt, where '1_010000' is the
    checkpoint's model identifier and step number.
    """
    # Split the filename at 'model_' and check if it results in a valid split
    split = filename.split("model_")
    if len(split) < 2:
        return 0  # Return 0 if the filename doesn't contain 'model_'

    # Extract the part after 'model_' and split it by the first underscore
    split1 = split[-1].split(".")[0]  # Remove the file extension '.pt'
    split_parts = split1.split("_")

    # Check if the split part has the expected number of components
    if len(split_parts) >= 2:
        try:
            # Return the numeric step part (e.g., 010000 from model_1_010000.pt)
            return int(split_parts[-1])
        except ValueError:
            return 0  # If there's a ValueError, return 0
    return 0  # Return 0 if the split part doesn't have the expected format


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    if main_checkpoint.split('/')[-1].startswith("latest"):
        prefix = 'latest_'
    else:
        prefix = ''
    filename = f"{prefix}ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
