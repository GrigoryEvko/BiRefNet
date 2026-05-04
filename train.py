import os
# Set CUDA allocator env vars BEFORE importing torch — once torch is imported the
# CUDA caching allocator may have already been initialized on some platforms.
# Track whether *we* set it; if the user already had a value (e.g.
# `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`) leave it alone and never pop.
_alloc_conf_was_user_set = 'PYTORCH_CUDA_ALLOC_CONF' in os.environ
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import datetime
import re
import argparse
import torch
import torch.nn as nn
import torch.optim as optim


def _torch_version_at_least(major: int, minor: int, patch: int = 0) -> bool:
    """Tolerant version compare; works on stable, +cu131 suffixes, and
    nightly stems like '2.11.0.dev20260101+cu131'."""
    m = re.match(r'(\d+)\.(\d+)(?:\.(\d+))?', torch.__version__)
    if not m:
        return False
    cur = (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))
    return cur >= (major, minor, patch)


if not _alloc_conf_was_user_set and not _torch_version_at_least(2, 5, 0):
    # Below 2.5.0 didn't support expandable_segments; clear the env var so we
    # don't poison the runtime with a value the allocator rejects. Only do
    # this when the value was set by us — otherwise we'd silently drop the
    # user's own configuration.
    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)

from birefnet_config import Config
from loss import PixLoss, ClsLoss
from dataset import MyData
from models.birefnet import BiRefNet
from utils import Logger, AverageMeter, set_seed, check_state_dict

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


parser = argparse.ArgumentParser(description='')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--ckpt_dir', default='ckpts/tmp', help='Temporary folder')
parser.add_argument('--dist', default=False, type=lambda x: x == 'True')
parser.add_argument('--use_accelerate', action='store_true', help='`accelerate launch --multi_gpu train.py --use_accelerate`. Use accelerate for training, good for FP16/BF16/...')
args = parser.parse_args()

config = Config()

if args.use_accelerate:
    from accelerate import Accelerator, utils
    mixed_precision = config.mixed_precision
    kwargs_handlers = [
            utils.InitProcessGroupKwargs(backend="nccl", timeout=datetime.timedelta(seconds=3600*10)),
            utils.DistributedDataParallelKwargs(find_unused_parameters=False),
            utils.GradScalerKwargs(backoff_factor=0.5),
    ]
    if mixed_precision == 'fp8':
        kwargs_handlers.append(utils.AORecipeKwargs())
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=1,
        kwargs_handlers=kwargs_handlers,
    )
    accelerator.print(accelerator.state)
    accelerator.print('backbone:', config.bb, ', freeze_bb:', config.freeze_bb)
    args.dist = False

# DDP
to_be_distributed = args.dist
if to_be_distributed:
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*10))
    device = int(os.environ["LOCAL_RANK"])
else:
    if args.use_accelerate:
        device = accelerator.local_process_index
    else:
        device = config.device

if config.rand_seed:
    set_seed(config.rand_seed + device)

epoch_st = 1
# make dir for ckpt
os.makedirs(args.ckpt_dir, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))
logger_loss_idx = 1

# log model and optimizer params
# logger.info("Model details:"); logger.info(model)
# if args.use_accelerate and accelerator.mixed_precision != 'no':
#     config.compile = False
logger.info("datasets: load_all={}, compile={}.".format(config.load_all, config.compile))
logger.info("Other hyperparameters:"); logger.info(args)
print('batch size:', config.batch_size)

from dataset import custom_collate_fn

def prepare_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, to_be_distributed=False, is_train=True):
    # Prepare dataloaders
    if to_be_distributed:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=False, sampler=DistributedSampler(dataset), drop_last=True, collate_fn=custom_collate_fn if is_train and config.dynamic_size else None
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=is_train, sampler=None, drop_last=True, collate_fn=custom_collate_fn if is_train and config.dynamic_size else None
        )


def init_data_loaders(to_be_distributed):
    # Prepare datasets
    train_loader = prepare_dataloader(
        MyData(datasets=config.training_set, data_size=None if config.dynamic_size else config.size, is_train=True),
        config.batch_size, to_be_distributed=to_be_distributed, is_train=True
    )
    print(len(train_loader), "batches of train dataloader {} have been created.".format(config.training_set))
    return train_loader


def init_models_optimizers(epochs, to_be_distributed):
    # Init models
    if config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=True and not os.path.isfile(str(args.resume)))
    else:
        print('Undefined model: {}.'.format(config.model))
        return None
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            state_dict = torch.load(args.resume, map_location='cpu', weights_only=True)
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            global epoch_st
            epoch_st = int(os.path.splitext(args.resume)[0].split('epoch_')[-1]) + 1
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    if not args.use_accelerate:
        if to_be_distributed:
            model = model.to(device)
            model = DDP(model, device_ids=[device])
        else:
            model = model.to(device)
    if config.compile:
        # 'max-autotune-no-cudagraphs' adds Triton kernel autotuning without
        # the cudagraphs that don't play well with DDP gradient hooks or
        # dynamic_size. 'default' is the safe fallback. 'reduce-overhead' /
        # 'max-autotune' use cudagraphs and need static shapes — wrong for
        # training. Picked via getattr so existing configs keep working.
        compile_mode = getattr(config, 'compile_mode', 'max-autotune-no-cudagraphs')
        model = torch.compile(model, mode=compile_mode)
    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')

    # Setting optimizer
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params=[p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=1e-2)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[lde if lde > 0 else epochs + lde + 1 for lde in config.lr_decay_epochs],
        gamma=config.lr_decay_rate
    )
    # logger.info("Optimizer details:"); logger.info(optimizer)

    return model, optimizer, lr_scheduler


class Trainer:
    def __init__(
        self, data_loaders, model_opt_lrsch,
    ):
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
        self.train_loader = data_loaders
        if args.use_accelerate:
            self.train_loader, self.model, self.optimizer = accelerator.prepare(self.train_loader, self.model, self.optimizer)
        if config.out_ref:
            # BCELoss is autocast-unsafe (raises in torch >= 2.x under bf16/fp16);
            # gdt prediction is a raw logit, so feed logits to BCEWithLogitsLoss.
            self.criterion_gdt = nn.BCEWithLogitsLoss()

        # Setting Losses
        self.pix_loss = PixLoss()
        self.cls_loss = ClsLoss()
        # Snapshot the original lambdas so finetune-window adjustments are
        # absolute (not cumulative across epochs).
        self._pix_lambdas_orig = dict(self.pix_loss.lambdas_pix_last)

        # Others
        self.loss_log = AverageMeter()

    def _train_batch(self, batch):
        if args.use_accelerate:
            inputs = batch[0]#.to(device)
            gts = batch[1]#.to(device)
            class_labels = batch[2]#.to(device)
        else:
            inputs = batch[0].to(device)
            gts = batch[1].to(device)
            class_labels = batch[2].to(device)
        self.optimizer.zero_grad()
        scaled_preds, class_preds_lst = self.model(inputs)
        if config.out_ref:
            (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
            loss_gdt = 0.
            for _gdt_pred, _gdt_label in zip(outs_gdt_pred, outs_gdt_label):
                # _gdt_pred is raw logits → feed straight to BCEWithLogitsLoss.
                # _gdt_label is laplacian(input)*mask_logit; sigmoid gives a soft
                # target in (0,1) that BCEWithLogitsLoss accepts directly.
                _gdt_pred = nn.functional.interpolate(
                    _gdt_pred, size=_gdt_label.shape[2:], mode='bilinear',
                    align_corners=bool(getattr(config, 'align_corners', True))
                )
                _gdt_label = _gdt_label.sigmoid()
                loss_gdt = loss_gdt + self.criterion_gdt(_gdt_pred, _gdt_label)
        if None in class_preds_lst:
            loss_cls = 0.
        else:
            loss_cls = self.cls_loss(class_preds_lst, class_labels)
            self.loss_dict['loss_cls'] = loss_cls.item()

        # Loss
        loss_pix, loss_dict_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1), pix_loss_lambda=1.0)
        self.loss_dict.update(loss_dict_pix)
        self.loss_dict['loss_pix'] = loss_pix.item()
        # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
        loss = loss_pix + loss_cls
        if config.out_ref:
            loss = loss + loss_gdt * 1.0

        self.loss_log.update(loss.item(), inputs.size(0))
        if args.use_accelerate:
            loss = loss / accelerator.gradient_accumulation_steps
            accelerator.backward(loss)
        else:
            loss.backward()
        self.optimizer.step()

    def train_epoch(self, epoch):
        global logger_loss_idx
        self.model.train()
        self.loss_dict = {}
        # Apply finetune-window lambda overrides as absolute multipliers of the
        # original snapshot. Doing it via `*= 0.9` on the live dict (the old code)
        # decayed geometrically every epoch — after 20 epochs `iou` was 0.5**20 of
        # its starting value, not 0.5x.
        if epoch > args.epochs + config.finetune_last_epochs:
            orig = self._pix_lambdas_orig
            current = self.pix_loss.lambdas_pix_last
            if config.task == 'Matting':
                current['mae'] = orig['mae'] * 1.0
                current['mse'] = orig['mse'] * 0.9
                current['ssim'] = orig['ssim'] * 0.9
            else:
                current['bce'] = orig['bce'] * 0.0
                current['ssim'] = orig['ssim'] * 1.0
                current['iou'] = orig['iou'] * 0.5
                current['mae'] = orig['mae'] * 0.9

        for batch_idx, batch in enumerate(self.train_loader):
            # with nullcontext if not args.use_accelerate or accelerator.gradient_accumulation_steps <= 1 else accelerator.accumulate(self.model):
            self._train_batch(batch)
            # Logger
            if (epoch < 2 and batch_idx < 100 and batch_idx % 20 == 0) or batch_idx % max(100, len(self.train_loader) / 100 // 100 * 100) == 0:
                info_progress = f'Epoch[{epoch}/{args.epochs}] Iter[{batch_idx}/{len(self.train_loader)}].'
                info_loss = 'Training Losses:'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += f' {loss_name}: {loss_value:.5g} |'
                logger.info(' '.join((info_progress, info_loss)))
        info_loss = f'@==Final== Epoch[{epoch}/{args.epochs}]  Training Loss: {self.loss_log.avg:.5g}  '
        logger.info(info_loss)

        self.lr_scheduler.step()
        return self.loss_log.avg


def main():

    trainer = Trainer(
        data_loaders=init_data_loaders(to_be_distributed),
        model_opt_lrsch=init_models_optimizers(args.epochs, to_be_distributed)
    )

    try:
        for epoch in range(epoch_st, args.epochs+1):
            train_loss = trainer.train_epoch(epoch)
            # Save checkpoint
            if epoch >= args.epochs - config.save_last and epoch % config.save_step == 0:
                if args.use_accelerate:
                    # Save once from main process; unwrap to drop accelerate's
                    # DDP/compile wrappers so the checkpoint is portable. Using
                    # trainer.model.state_dict() on every rank wrote the same
                    # file N times and corrupted under concurrent fsync.
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(trainer.model)
                        accelerator.save(
                            unwrapped.state_dict(),
                            os.path.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch)),
                        )
                else:
                    state_dict = trainer.model.module.state_dict() if to_be_distributed else trainer.model.state_dict()
                    # Vanilla DDP: rank 0 only — gradient ranks shouldn't race
                    # on the same file.
                    if not to_be_distributed or torch.distributed.get_rank() == 0:
                        torch.save(state_dict, os.path.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch)))
    finally:
        # destroy_process_group must run on Ctrl-C / unexpected exit too —
        # otherwise NCCL leaves zombie comms in /dev/shm and the next launch
        # gets EADDRINUSE on the rendezvous port.
        if to_be_distributed:
            destroy_process_group()


if __name__ == '__main__':
    main()
