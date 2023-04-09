import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from omegaconf import OmegaConf

# class ImageLogger(Callback):
#     def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
#                  rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
#                  log_images_kwargs=None):
#         super().__init__()
#         self.rescale = rescale
#         self.batch_freq = batch_frequency
#         self.max_images = max_images
#         if not increase_log_steps:
#             self.log_steps = [self.batch_freq]
#         self.clamp = clamp
#         self.disabled = disabled
#         self.log_on_batch_idx = log_on_batch_idx
#         self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
#         self.log_first_step = log_first_step

#     @rank_zero_only
#     def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
#         root = os.path.join(save_dir, "image_log", split)
#         for k in images:
#             grid = torchvision.utils.make_grid(images[k], nrow=4)
#             if self.rescale:
#                 grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
#             grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
#             grid = grid.numpy()
#             grid = (grid * 255).astype(np.uint8)
#             filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
#             path = os.path.join(root, filename)
#             os.makedirs(os.path.split(path)[0], exist_ok=True)
#             Image.fromarray(grid).save(path)

#     def log_img(self, pl_module, batch, batch_idx, split="train"):
#         check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
#         if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
#                 hasattr(pl_module, "log_images") and
#                 callable(pl_module.log_images) and
#                 self.max_images > 0):
#             logger = type(pl_module.logger)

#             is_train = pl_module.training
#             if is_train:
#                 pl_module.eval()

#             with torch.no_grad():
#                 images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

#             for k in images:
#                 N = min(images[k].shape[0], self.max_images)
#                 images[k] = images[k][:N]
#                 if isinstance(images[k], torch.Tensor):
#                     images[k] = images[k].detach().cpu()
#                     if self.clamp:
#                         images[k] = torch.clamp(images[k], -1., 1.)

#             self.log_local(pl_module.logger.save_dir, split, images,
#                            pl_module.global_step, pl_module.current_epoch, batch_idx)

#             if is_train:
#                 pl_module.train()

#     def check_frequency(self, check_idx):
#         return check_idx % self.batch_freq == 0

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         if not self.disabled:
#             self.log_img(pl_module, batch, batch_idx, split="train")


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not self.disabled and pl_module.global_step > 0:
        #     self.log_img(pl_module, batch, batch_idx, split="val")
        # if hasattr(pl_module, 'calibrate_grad_norm'):
        #     if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
        #         self.log_gradients(trainer, pl_module, batch_idx=batch_idx)
        pass