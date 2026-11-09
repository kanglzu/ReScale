from __future__ import annotations
import hashlib
import json
import os
import platform
import random
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, get_worker_info
from rescale.data.split import load_manifest, load_splits, get_sample_dict
from rescale.data.transforms import get_train_transforms, get_val_transforms
from rescale.engine.loss import build_loss
from rescale.evaluation.metrics import EMATracker
from rescale.engine.logger import TrainingLogger
from rescale.engine.naming import fold_output_name
from torch.nn import functional as F
from rescale.engine.loss import DiceCELoss
from rescale.engine.amp import make_cuda_grad_scaler
from rescale.recursion.sampling import one_roi_training_enabled, select_one_roi_per_image
from rescale.build import build_model

FIRST_CHECKPOINT_EPOCH = 50  # One-based, matching the paper experiments.

def _finite_json(value):
    """Represent undefined scientific metrics as JSON null, including nesting."""
    if isinstance(value, dict):
        return {key: _finite_json(item) for (key, item) in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value

def _masked_segmentation_loss(logits, targets, valid, *, ce_weight=0.5, smooth=1e-05, reduction='mean'):
    """Dice+CE excluding crop padding in both foreground and background."""
    logits = logits.float()
    if valid.ndim == 4:
        valid = valid[:, 0]
    valid = valid.float().clamp(0.0, 1.0)
    if tuple(targets.shape[-2:]) != tuple(logits.shape[-2:]):
        targets = F.interpolate(targets[:, None].float(), size=logits.shape[-2:], mode='nearest')[:, 0].long()
        valid = F.interpolate(valid[:, None], size=logits.shape[-2:], mode='nearest')[:, 0]
    targets = torch.where(valid > 0, targets, torch.zeros_like(targets)).long()
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    weight = valid[:, None]
    intersection = (probs * one_hot * weight).sum(dim=(2, 3))
    union = ((probs + one_hot) * weight).sum(dim=(2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + smooth) / (union + smooth))[:, 1:].mean(dim=1)
    ce = F.cross_entropy(logits, targets, reduction='none')
    ce_loss = (ce * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1.0)
    losses = (1.0 - ce_weight) * dice_loss + ce_weight * ce_loss
    if reduction == 'none':
        return losses
    if reduction == 'mean':
        return losses.mean()
    raise ValueError('Crop loss reduction must be mean or none')

def _teacher_forcing_options(cfg):
    """Validate the optional training warmup before building any data or model."""
    options = cfg.get('innovations', {}).get('recursion', {}).get('teacher_forcing', {})
    if not isinstance(options, dict):
        raise ValueError('teacher_forcing must be a configuration mapping')
    options = dict(options)
    options.setdefault('enabled', False)
    options.setdefault('epochs', 50)
    options.setdefault('min_area', 1)
    if not isinstance(options['enabled'], bool):
        raise ValueError('teacher_forcing.enabled must be true or false')
    for (name, minimum) in (('epochs', 0), ('min_area', 1)):
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'teacher_forcing.{name} must be an integer >= {minimum}')
    return options


def _crop_source(cfg):
    source = cfg.get('innovations', {}).get('recursion', {}).get('crop_source', 'native')
    if source not in ('native', 'resized'):
        raise ValueError('recursion.crop_source must be native or resized')
    return source

class ReScaleTrainer:

    def _initialize_training(self, cfg: dict):
        self.cfg = cfg
        self.fold = cfg['training']['fold']
        self.grad_accum_steps = max(int(cfg['training'].get('grad_accum_steps', 1)), 1)
        self._setup_distributed()
        base_seed = cfg['training'].get('seed', 42)
        self._set_seed(base_seed)
        self._base_seed = base_seed
        if self.is_main:
            print(f'[Seed] base={base_seed}  rank_seeds={base_seed}..{base_seed + self.world_size - 1}')
        save_dir = Path(cfg['output']['save_dir'])
        self.output_dir = save_dir / fold_output_name(cfg, self.fold)
        self.repro_manifest_path = self.output_dir / 'repro_manifest.json'
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if self._use_ddp:
            dist.barrier()
        self.num_epochs = cfg['training']['num_epochs']
        self.run_epochs = cfg['training'].get('stop_after_epochs', self.num_epochs)
        if type(self.run_epochs) is not int or not 1 <= self.run_epochs <= self.num_epochs:
            raise ValueError('stop_after_epochs must be an integer in [1, num_epochs]')
        self.current_epoch = 0
        metrics_cfg = cfg.get('metrics', {})
        self._val_metrics: Set[str] = set(metrics_cfg.get('val', ['dice', 'iou']))
        self._val_metrics.add('dice')
        self._val_hd95_interval: int = metrics_cfg.get('val_hd95_interval', 5)
        self._val_interval: int = max(int(metrics_cfg.get('val_interval', 1)), 1)
        self._build_data()
        self._build_model()
        if self._use_ddp:
            self._set_seed(self._base_seed + self.rank)
        self._build_optimizer()
        self.loss_fn = self._build_loss()
        self.ema = EMATracker(alpha=cfg['training'].get('ema_alpha', 0.9))
        if self.is_main:
            self.logger = TrainingLogger(str(self.output_dir), self.num_epochs)
        self.amp_enabled = cfg['training'].get('amp', True) and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        if self.is_main:
            batch_size = cfg['training']['batch_size']
            print(f'[Train] batch/rank={batch_size}  accum={self.grad_accum_steps}  effective_global_batch={batch_size * self.world_size * self.grad_accum_steps}  amp={self.amp_enabled}')
        self.min_delta = cfg['training'].get('min_delta', 0.0001)
        self._best_epoch = -1
        self._best_es_value = -1.0

    def _setup_distributed(self) -> None:
        if 'LOCAL_RANK' in os.environ:
            self.local_rank = int(os.environ['LOCAL_RANK'])
            dist.init_process_group(backend='nccl')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self._use_ddp = self.world_size > 1
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1
            self._use_ddp = False
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_main = self.rank == 0
        if self.is_main:
            mode = f'DDP  world_size={self.world_size}' if self._use_ddp else 'single-GPU'
            print(f'[Distributed] {mode}  device={self.device}')

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

    def _make_loader_generator(self, offset: int) -> torch.Generator:
        """Create a fixed-seed DataLoader generator, independent of the global RNG."""
        g = torch.Generator()
        g.manual_seed(int(self._base_seed) + self.rank * 1000 + offset)
        return g

    @staticmethod
    def _seed_worker_base(worker_id: int) -> None:
        """Seed DataLoader worker RNGs from PyTorch's per-worker initial seed."""
        del worker_id
        worker_seed = torch.initial_seed() % 2 ** 32
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        info = get_worker_info()
        if info is not None:
            ds: Any = info.dataset
            while not hasattr(ds, 'set_worker_seed') and hasattr(ds, 'dataset'):
                ds = ds.dataset
            if hasattr(ds, 'set_worker_seed'):
                ds.set_worker_seed(worker_seed)

    def _normalized_config(self) -> Dict[str, Any]:
        return deepcopy(self.cfg)

    def _build_repro_manifest(self) -> Dict[str, Any]:
        cfg_blob = json.dumps(self._normalized_config(), sort_keys=True, ensure_ascii=True, separators=(',', ':'))
        try:
            import torchvision
            tv_ver = torchvision.__version__
        except Exception:
            tv_ver = None
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        return {'config_hash': hashlib.sha256(cfg_blob.encode()).hexdigest(), 'config': self._normalized_config(), 'seed': int(self._base_seed), 'model_name': self.cfg['model']['name'], 'world_size': self.world_size, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu_names': gpu_names, 'python_version': platform.python_version(), 'torch_version': torch.__version__, 'torchvision_version': tv_ver, 'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(), 'argv': sys.argv}

    def _prepare_reproducibility_run(self) -> None:
        manifest = self._build_repro_manifest()
        if self.is_main:
            with open(self.repro_manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, sort_keys=True)
            print(f'[Reproducibility] manifest saved -> {self.repro_manifest_path}')
        if self._use_ddp:
            dist.barrier()

    def _sync_phase(self) -> None:
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)


    def _phase_time_start(self) -> float:
        self._sync_phase()
        return time.perf_counter()

    def _phase_time_end(self, t0: float) -> float:
        self._sync_phase()
        return time.perf_counter() - t0




    @property
    def _network_module(self) -> nn.Module:
        return self.network.module if self._use_ddp else self.network

    def _reduce_mean(self, val: float) -> float:
        if not self._use_ddp:
            return val
        t = torch.tensor(val, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.item() / self.world_size

    def _broadcast_from_main(self, value: Any) -> Any:
        if not self._use_ddp:
            return value
        payload = [value if self.is_main else None]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def _make_epoch_decision(self, val_metrics: Dict) -> Dict[str, Any]:
        """Select a checkpoint by validation Dice."""
        es_value = val_metrics.get('dice')
        if es_value is None:
            raise RuntimeError('Validation Dice is required for checkpoint selection.')
        is_best = es_value - self._best_es_value > self.min_delta
        best_es = es_value if is_best else self._best_es_value
        best_ep = self.current_epoch if is_best else self._best_epoch
        return {'is_best': is_best, 'best_es_value': best_es, 'best_epoch': best_ep}

    def _build_loaders(self) -> None:
        data_cfg = self.cfg['data']
        manifest = load_manifest(data_cfg['manifest_json'])
        sample_dict = get_sample_dict(manifest)
        num_channels = len(manifest.get('channel_names', {'0': ''}))
        image_size = data_cfg.get('image_size', [256, 256])
        seed = self.cfg['training'].get('seed', 42)
        seeded_transforms = seed is not None
        (train_ids, val_ids) = load_splits(data_cfg['splits_json'], self.fold)
        (dataset_cls, train_builder, val_builder, collate_fn) = self._data_components()
        train_transform = train_builder(image_size, num_channels) if not seeded_transforms else None
        val_transform = val_builder(image_size, num_channels) if not seeded_transforms else None
        train_transform_factory = partial(train_builder, image_size, num_channels) if seeded_transforms else None
        val_transform_factory = partial(val_builder, image_size, num_channels) if seeded_transforms else None
        if data_cfg.get('use_landmarks', False):
            raise ValueError('ReScale expects image and mask pairs')
        train_ds = dataset_cls(sample_ids=train_ids, sample_dict=sample_dict, transform=train_transform, transform_factory=train_transform_factory, num_channels=num_channels, mask_mode=data_cfg.get('mask_mode', 'binary'), num_classes=self.cfg['model']['num_classes'])
        val_ds = dataset_cls(sample_ids=val_ids, sample_dict=sample_dict, transform=val_transform, transform_factory=val_transform_factory, num_channels=num_channels, mask_mode=data_cfg.get('mask_mode', 'binary'), num_classes=self.cfg['model']['num_classes'])
        num_workers = data_cfg.get('num_workers', 4)
        pin_memory = data_cfg.get('pin_memory', True) and self.device.type == 'cuda'
        batch_size = self.cfg['training']['batch_size']
        val_batch_size = data_cfg.get('val_batch_size')
        if val_batch_size is None:
            val_batch_size = batch_size
        persistent_workers = bool(data_cfg.get('persistent_workers', False))
        if self._use_ddp:
            train_sampler: Optional[DistributedSampler] = DistributedSampler(train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True, seed=seed, drop_last=True)
            val_rank_indices = list(range(self.rank, len(val_ds), self.world_size))
            val_ds_for_loader = torch.utils.data.Subset(val_ds, val_rank_indices)
            train_shuffle = False
        else:
            train_sampler = None
            val_ds_for_loader = val_ds
            train_shuffle = True
        train_generator = self._make_loader_generator(offset=101)
        val_generator = self._make_loader_generator(offset=202)
        if seeded_transforms and num_workers == 0:
            train_ds.set_worker_seed(int(seed) + self.rank * 1000 + 11)
            val_ds.set_worker_seed(int(seed) + self.rank * 1000 + 22)
        prefetch = 2 if num_workers > 0 else None
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=train_shuffle, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, generator=train_generator, worker_init_fn=self._seed_data_worker, persistent_workers=persistent_workers if num_workers > 0 else False, prefetch_factor=prefetch, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_ds_for_loader, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, generator=val_generator, worker_init_fn=self._seed_data_worker, persistent_workers=persistent_workers if num_workers > 0 else False, prefetch_factor=prefetch, collate_fn=collate_fn)
        self.num_classes = manifest.get('num_classes', self.cfg['model']['num_classes'])
        if self.is_main:
            print(f'[Data] fold={self.fold}  train={len(train_ids)}  val={len(val_ids)}')

    def _make_param_layout_contiguous(self) -> None:
        with torch.no_grad():
            for p in self.network.parameters():
                if not p.is_contiguous():
                    p.data = p.data.contiguous()
            for b in self.network.buffers():
                if b.is_floating_point() and (not b.is_contiguous()):
                    b.data = b.data.contiguous()

    def _build_model(self) -> None:
        model_cfg = self.cfg
        self.network = build_model(model_cfg).to(self.device)
        if self.is_main:
            total = sum((p.numel() for p in self.network.parameters()))
            trainable = sum((p.numel() for p in self.network.parameters() if p.requires_grad))
            print(f"[Model] {self.cfg['model']['name']}  params={total / 1000000.0:.2f}M  trainable={trainable / 1000000.0:.2f}M")
            if self._use_ddp:
                print('[FLOPs] Skipped torchinfo forward in DDP pre-wrap to avoid rank-specific side effects.')
            else:
                try:
                    from torchinfo import summary
                    img_size = self.cfg['data']['image_size']
                    in_ch = self.cfg['model']['in_channels']
                    macs = summary(self.network, input_size=(1, in_ch, *img_size), verbose=0).total_mult_adds
                    print(f'[FLOPs] MACs={macs / 1000000000.0:.2f}G  FLOPs~{2 * macs / 1000000000.0:.2f}G  (image_size={img_size})')
                except Exception as e:
                    print(f'[FLOPs] Could not compute: {e}')
        if self._use_ddp:
            self._make_param_layout_contiguous()
            dist.barrier()
            model_name = str(self.cfg['model'].get('name', ''))
            if 'find_unused_parameters' in self.cfg['model']:
                has_unused = bool(self.cfg['model']['find_unused_parameters'])
            else:
                has_unused = not self.cfg['model'].get('deep_supervision', False)
            self.network = DDP(self.network, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=has_unused, broadcast_buffers=bool(self.cfg['model'].get('broadcast_buffers', True)))

    def _build_loss(self):
        return build_loss(self.cfg)

    def _compute_train_loss(self, images: torch.Tensor, outputs, masks: torch.Tensor) -> torch.Tensor:
        """Compute training loss. Subclasses may add aux losses."""
        return self.loss_fn(outputs, masks)

    def _build_optimizer(self) -> None:
        tr = self.cfg['training']
        lr = tr.get('initial_lr', 0.0001)
        wd = tr.get('weight_decay', 0.05)
        opt_name = tr.get('optimizer', 'adamw').lower()
        if opt_name != 'adamw':
            raise ValueError('ReScale uses AdamW')
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr, weight_decay=wd, eps=1e-05, betas=(0.9, 0.999))
        sched_name = tr.get('lr_schedule', 'cosine').lower()
        warmup_epochs = int(tr.get('warmup_epochs', 0))
        T_max = max(self.num_epochs - warmup_epochs, 1)
        if sched_name != 'cosine':
            raise ValueError('ReScale uses cosine learning-rate decay')
        main = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=tr.get('eta_min', 1e-06))
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda ep: float(ep + 1) / float(warmup_epochs) if ep < warmup_epochs else 1.0)
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])
        else:
            self.scheduler = main

    @staticmethod
    def _checkpoint_model_state(ckpt):
        if isinstance(ckpt, dict):
            return ckpt.get('model_state', ckpt.get('state_dict', ckpt))
        return ckpt

    def _optimize_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Run one full pass through the training set; returns (mean_loss, timing_stats)."""
        self.network.train()
        if self._use_ddp:
            assert isinstance(self.train_loader.sampler, DistributedSampler)
            self.train_loader.sampler.set_epoch(self.current_epoch)
        grad_clip = self.cfg['training'].get('grad_clip', 12.0)
        num_steps = len(self.train_loader)
        if num_steps == 0:
            raise RuntimeError('train_loader is empty after batching.')
        total_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        timing = {}
        self.optimizer.zero_grad(set_to_none=True)
        for (step, batch) in enumerate(self.train_loader):
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            should_step = (step + 1) % self.grad_accum_steps == 0 or step + 1 == num_steps
            try:
                loss = self._backward_train_batch(images, masks, batch, should_sync=should_step)
                total_loss += loss.detach().double()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    raise RuntimeError('CUDA OOM during forward/backward. Reduce batch_size, increase grad_accum_steps to fit the per-GPU memory budget.') from e
                raise
            if should_step:
                self.scaler.unscale_(self.optimizer)
                self._normalize_accumulated_gradients()
                nn.utils.clip_grad_norm_(self.network.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        return (self._reduce_mean(total_loss.item() / max(num_steps, 1)), timing)

    def _validate_epoch(self, compute_hd: bool=False) -> Dict:
        """Run validation; only emits metrics listed in cfg.metrics.val."""
        result = self._run_eval(self.val_loader, compute_hd=compute_hd, distributed=self._use_ddp)
        out: Dict = {'val_loss': result['loss'], 'pseudo_dice': result['pseudo_dice']}
        if 'dice' in self._val_metrics:
            out['dice'] = result['dice']
        if 'iou' in self._val_metrics:
            out['iou'] = result['iou']
        out['hd95'] = result['hd95'] if compute_hd and 'hd95' in self._val_metrics else None
        for (name, metrics) in result['extra_metrics'].items():
            out[f'{name}_dice'] = metrics['dice']
            out[f'{name}_iou'] = metrics['iou']
        out['roi_count_per_image'] = result['roi_count_per_image']
        return out

    def _fit(self) -> None:
        """Execute training and validation."""
        best_ckpt = self.output_dir / 'checkpoint_best.pth'
        self._prepare_reproducibility_run()
        if self.is_main and self.current_epoch == 0:
            val_m = sorted({'pseudo_dice', 'ema_pseudo_dice'} | self._val_metrics)
            with open(self.logger.log_file, 'w') as f:
                f.write(f"ReScale  model={self.cfg['model']['name']}  fold={self.fold}  device={self.device}  world_size={self.world_size}\nval_metrics={val_m}  val_hd95_interval={self._val_hd95_interval}  best_metric=val_dice\n{'=' * 80}\n")
        for epoch in range(self.current_epoch, self.run_epochs):
            self.current_epoch = epoch
            epoch_t0 = self._phase_time_start()
            train_t0 = self._phase_time_start()
            (train_loss, timing) = self.train_one_epoch()
            timing['train'] = self._phase_time_end(train_t0)
            do_val = epoch % self._val_interval == 0 or epoch + 1 == self.run_epochs
            if do_val:
                compute_hd = 'hd95' in self._val_metrics and epoch % self._val_hd95_interval == 0
                val_t0 = self._phase_time_start()
                val_metrics = self.validate_one_epoch(compute_hd=compute_hd)
                timing['val'] = self._phase_time_end(val_t0)
                val_loss = val_metrics['val_loss']
                pseudo_dice = val_metrics['pseudo_dice']
                if val_metrics.get('dice') is None:
                    raise RuntimeError('Validation Dice is required for checkpoint selection.')
                ema_pseudo_dice = self.ema.update(pseudo_dice)
            else:
                (val_metrics, val_loss, pseudo_dice) = ({}, None, None)
                ema_pseudo_dice = self.ema.value
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
            epoch_time = self._phase_time_end(epoch_t0)
            if do_val and epoch + 1 >= FIRST_CHECKPOINT_EPOCH:
                assert ema_pseudo_dice is not None
                decision = self._make_epoch_decision(val_metrics) if self.is_main or not self._use_ddp else None
                decision = self._broadcast_from_main(decision)
                self._best_es_value = decision['best_es_value']
                self._best_epoch = decision['best_epoch']
                is_best = decision['is_best']
            else:
                is_best = False
            if is_best:
                self.save_checkpoint(str(best_ckpt))
            if self._use_ddp:
                dist.barrier()
            (global_loss, focused_loss) = (None, None)
            if hasattr(self, 'pop_epoch_round_losses'):
                (global_loss, focused_loss) = self.pop_epoch_round_losses()
            if self.is_main:
                self.logger.log_epoch(epoch=epoch, train_loss=train_loss, val_loss=val_loss, pseudo_dice=pseudo_dice, ema_pseudo_dice=ema_pseudo_dice, dice=val_metrics.get('dice'), iou=val_metrics.get('iou'), lr=lr, epoch_time=epoch_time, hd95=val_metrics.get('hd95'), is_best=is_best, timing=timing, global_loss=global_loss, focused_loss=focused_loss, base_dice=val_metrics.get('base_dice'), base_iou=val_metrics.get('base_iou'), roi_count_per_image=val_metrics.get('roi_count_per_image'), train_roi_per_image=getattr(self, '_train_roi_per_image', None))
                self.logger.print_epoch_summary(epoch)
        if self.is_main and self._best_epoch >= 0:
            print(f'\n[Done] Best val dice={self._best_es_value:.4f} at epoch {self._best_epoch + 1}.  Checkpoint: {best_ckpt}')
        if self._use_ddp:
            dist.barrier()
        final_metrics: Dict = {}
        if self._best_epoch >= 0:
            if self.is_main:
                print(f'[Final Eval] Reloading best checkpoint for full metric eval -> {best_ckpt}')
            ckpt = torch.load(str(best_ckpt), map_location=self.device)
            self._network_module.load_state_dict(self._checkpoint_model_state(ckpt))
            compute_hd = 'hd95' in self._val_metrics
            final_metrics = self.validate_checkpoint(compute_hd=compute_hd)
            if self.is_main:
                parts = []
                for key in ('dice', 'iou', 'hd95'):
                    v = final_metrics.get(key)
                    if v is not None:
                        parts.append(f'{key}={v:.4f}' if key != 'hd95' else f'{key}={v:.2f}')
                print('[Final Eval] ' + '  '.join(parts))
        if self.is_main:
            self._write_results(final_metrics=final_metrics)
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def _save_result_summary(self, final_metrics: Optional[Dict]=None) -> None:
        """Write per-fold results.json (rank 0 only). Used by scripts/summarize_cv.py."""
        b = self._best_epoch

        def _get(key):
            vals = self.logger._data.get(key, [])
            return vals[b] if 0 <= b < len(vals) and vals[b] is not None else None
        val_result: Dict = {'ema_pseudo_dice': _get('ema_pseudo_dice')}
        metric_keys = ('dice', 'iou', 'hd95', 'base_dice', 'base_iou')
        for key in metric_keys:
            v = _get(key)
            if v is not None:
                val_result[key] = v
        if 'dice' in val_result and 'base_dice' in val_result:
            val_result['refinement_gain_pp'] = 100.0 * (val_result['dice'] - val_result['base_dice'])
        checkpoint_eval: Dict = {}
        for key in metric_keys:
            v = (final_metrics or {}).get(key)
            if v is not None:
                checkpoint_eval[key] = v
                if key in ('dice', 'iou', 'hd95'):
                    val_result.setdefault(key, v)
        if 'dice' in checkpoint_eval and 'base_dice' in checkpoint_eval:
            checkpoint_eval['refinement_gain_pp'] = 100.0 * (checkpoint_eval['dice'] - checkpoint_eval['base_dice'])
        out_path = self.output_dir / 'results.json'
        with open(out_path, 'w') as f:
            payload = {'model': self.cfg['model']['name'], 'fold': self.fold, 'best_epoch': b + 1 if b >= 0 else None, 'selection_metric': 'val.dice', 'val': val_result}
            if checkpoint_eval:
                payload['best_checkpoint_eval'] = checkpoint_eval
            if 'stop_after_epochs' in self.cfg['training']:
                payload['budget'] = dict(scheduled_epochs=self.num_epochs, executed_epochs=self.current_epoch + 1, stop_after_epochs=self.run_epochs, scope='training_prefix')
            json.dump(payload, f, indent=2)
        print(f'[Results] Saved -> {out_path}')

    def __init__(self, cfg):
        cfg = deepcopy(cfg)
        from rescale.utils.runtime import configure_cpu_threads
        configure_cpu_threads(cfg)
        if int(cfg['model']['num_classes']) < 2:
            raise ValueError('RecursiveSegTrainer requires background and foreground logits')
        cfg['model'].setdefault('input_size', cfg['data']['image_size'])
        if tuple(cfg['model']['input_size']) != tuple(cfg['data']['image_size']):
            raise ValueError('model.input_size must match data.image_size')
        _teacher_forcing_options(cfg)
        one_roi_training_enabled(cfg)
        _crop_source(cfg)
        cfg['model'].setdefault('find_unused_parameters', True)
        if cfg['model'].get('broadcast_buffers', False):
            raise ValueError('RecursiveSegTrainer requires model.broadcast_buffers=false')
        cfg['model']['broadcast_buffers'] = False
        self._initialize_training(cfg)
        dtype_name = str(cfg['training'].get('amp_dtype', 'fp16')).lower()
        choices = {'fp16': torch.float16, 'float16': torch.float16, 'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16}
        if dtype_name not in choices:
            raise ValueError('training.amp_dtype must be fp16 or bf16')
        self.amp_dtype = choices[dtype_name]
        if self.amp_enabled and self.amp_dtype == torch.bfloat16 and (not torch.cuda.is_bf16_supported()):
            raise ValueError('BF16 AMP was requested but this CUDA device does not support BF16')
        self.scaler = make_cuda_grad_scaler(enabled=self.amp_enabled and self.amp_dtype == torch.float16)
        import cv2
        print(f'[RecursiveCPU] rank={self.rank} torch_threads={torch.get_num_threads()} opencv_threads={cv2.getNumThreads()}', flush=True)
        print(f'[RecursiveSeg] crop_source={_crop_source(self.cfg)}', flush=True)
        print(f'[RecursiveSeg] train_one_roi_per_image={one_roi_training_enabled(self.cfg)}', flush=True)

    def _teacher_forcing_active(self):
        """Only the training proposal source changes during these first epochs."""
        module = self._network_module
        options = _teacher_forcing_options(self.cfg)
        return bool(module.recursion_enabled) and int(module.rounds) == 2 and options['enabled'] and (self.current_epoch < options['epochs'])

    def _autocast_context(self):
        return torch.autocast(device_type=self.device.type, dtype=getattr(self, 'amp_dtype', torch.float16), enabled=self.amp_enabled)

    def _data_components(self):
        from rescale.data.resized_dataset import RecursiveResizedSegDataset
        c = self.cfg['data'].get('recursive_cache', {})
        workers = max(1, int(self.cfg['data'].get('num_workers', 0)))
        enabled = c.get('enabled', False)
        dataset = partial(RecursiveResizedSegDataset, validation_memory_mb=c.get('validation_memory_mb', 0) / workers if enabled else 0, training_memory_mb=c.get('training_memory_mb', 0) / workers if enabled else 0, shared_training_memory_mb=c.get('training_memory_mb', 0) if enabled and c.get('share_training_resize', False) else 0, training_input_size=self.cfg['data']['image_size'])
        return (dataset, get_train_transforms, get_val_transforms, None)

    @staticmethod
    def _seed_data_worker(worker_id):
        ReScaleTrainer._seed_worker_base(worker_id)
        import cv2
        cv2.setNumThreads(1)

    def _build_data(self):
        self._shared_cache = None
        self._build_loaders()
        dataset = self.train_loader.dataset
        dataset.enable_training_cache()
        c = self.cfg['data'].get('recursive_cache', {})
        if c.get('enabled', False) and c.get('prewarm_training_resizes', False):
            dataset.prewarm_training_resizes(seed=self._base_seed + self.rank * 1000 + 11)


    def _normalize_accumulated_gradients(self):
        """Average the accumulated per-image sums once per optimizer update."""
        image_count = getattr(self, '_accumulated_image_count', 0)
        if image_count <= 0:
            raise RuntimeError('No training images were accumulated before optimizer step')
        if self._use_ddp:
            count = torch.tensor(float(image_count), device=self.device)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            factor = dist.get_world_size() / count
        else:
            factor = 1.0 / image_count
        gradients = [parameter.grad for parameter in self.network.parameters() if parameter.grad is not None]
        torch._foreach_mul_(gradients, factor)
        self._accumulated_image_count = 0

    def _crop_loss(self, outputs, targets, valid, reduction='mean'):
        ce_weight = float(self.cfg['training'].get('loss_ce_weight', 0.5))
        if not isinstance(outputs, (tuple, list)):
            return _masked_segmentation_loss(outputs, targets, valid, ce_weight=ce_weight, reduction=reduction)
        weights = getattr(self.loss_fn, 'weights', None)
        if weights is None:
            raise ValueError('Multiple segmentation heads require the configured deep-supervision loss')
        return sum((weights[min(index, len(weights) - 1)] * _masked_segmentation_loss(pred, targets, valid, ce_weight=ce_weight, reduction=reduction) for (index, pred) in enumerate(outputs)))

    def _select_training_rois(self, boxes, batch_indices, batch):
        sample_ids = batch.get('id')
        if sample_ids is None or len(sample_ids) != batch['image'].shape[0]:
            raise ValueError("One-ROI training requires the dataset sample IDs in batch['id']")
        return select_one_roi_per_image(boxes, batch_indices, sample_ids, seed=self.cfg['training'].get('seed', 42), fold=self.cfg['training'].get('fold', 0), epoch=self.current_epoch)

    def _backward_train_batch(self, images, masks, batch, *, should_sync):
        module = self._network_module
        batch_size = images.shape[0]
        recursive = bool(module.recursion_enabled) and int(module.rounds) == 2
        teacher_forcing = recursive and self._teacher_forcing_active()
        sample_one_roi = recursive and one_roi_training_enabled(self.cfg)
        sync_context = self.network.no_sync if self._use_ddp and (recursive or not should_sync) else nullcontext
        with sync_context():
            with self._autocast_context():
                outputs = self.network(images)
            loss = self._compute_train_loss(images, outputs, masks)
            if recursive and (not teacher_forcing):
                with torch.no_grad():
                    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                    (boxes, batch_indices) = module.make_rois(logits.detach())
            self.scaler.scale(loss * batch_size).backward()
            first_loss = loss.detach()
        del outputs, loss
        if recursive and (not teacher_forcing):
            del logits
        module.clear_aux_outputs()
        self._accumulated_image_count = getattr(self, '_accumulated_image_count', 0) + batch_size
        second_loss_sum = first_loss.new_zeros(())
        roi_counts = torch.zeros(batch_size, device=images.device, dtype=torch.long)
        roi_count = 0
        if recursive:
            if teacher_forcing:
                from rescale.recursion.geometry import lesion_boxes_from_labels
                options = _teacher_forcing_options(self.cfg)
                (boxes, batch_indices) = lesion_boxes_from_labels(masks, min_area=options['min_area'], margin=module.margin, backend=module.backend)
            if sample_one_roi:
                selected = self._select_training_rois(boxes, batch_indices, batch)
                (boxes, batch_indices) = (boxes[selected], batch_indices[selected])
            roi_count = boxes.shape[0]
            if roi_count == 0:
                raise ValueError('Recursive proposals must include a fallback crop for every image')
            roi_counts.scatter_add_(0, batch_indices, torch.ones_like(batch_indices))
            roi_weights = roi_counts[batch_indices].float().reciprocal()
            crop_batch_size = int(module.crop_batch_size)
            if crop_batch_size < 1:
                raise ValueError('crop_batch_size must be positive')
            crop_dtype = getattr(self, 'amp_dtype', torch.float16) if self.amp_enabled else images.dtype
            for start in range(0, roi_count, crop_batch_size):
                end = min(start + crop_batch_size, roi_count)
                last_chunk = end == roi_count
                sync_context = self.network.no_sync if self._use_ddp and not (should_sync and last_chunk) else nullcontext
                with sync_context():
                    with torch.no_grad():
                        (crop_images, crop_targets, crop_valid) = module.crop_resized(images, boxes[start:end], masks=masks, output_dtype=crop_dtype, batch_indices=batch_indices[start:end])
                        scale = module.crop_scale(boxes[start:end])
                    with self._autocast_context():
                        outputs = self.network(crop_images, scale=scale)
                    crop_losses = self._crop_loss(outputs, crop_targets, crop_valid, reduction='none')
                    chunk_loss = (crop_losses * roi_weights[start:end]).sum()
                    weighted_loss = float(module.loss_weight) * chunk_loss
                    self.scaler.scale(weighted_loss).backward()
                    second_loss_sum += chunk_loss.detach()
                del outputs, crop_images, crop_targets, crop_valid, scale
                del crop_losses, chunk_loss, weighted_loss
                module.clear_aux_outputs()
        if hasattr(self, '_round_loss_sums'):
            self._round_loss_sums += torch.stack((first_loss * batch_size, second_loss_sum, first_loss.new_tensor(batch_size), first_loss.new_tensor(roi_count))).double()
        module.clear_aux_outputs()
        return first_loss + float(module.loss_weight) * second_loss_sum / batch_size

    def train_one_epoch(self):
        teacher_forcing = self._teacher_forcing_active()
        dataset = getattr(self.train_loader, 'dataset', None)
        dataset.enable_training_cache()
        phase = 'teacher_forcing' if teacher_forcing else 'prediction' if self._network_module.recursion_enabled and self._network_module.rounds == 2 else 'disabled'
        if self.is_main and _teacher_forcing_options(self.cfg)['enabled']:
            print(f'[RecursiveSeg] epoch {self.current_epoch + 1}: {phase}', flush=True)
        self._accumulated_image_count = 0
        self._round_loss_sums = torch.zeros(4, device=self.device, dtype=torch.float64)
        (_, timing) = self._optimize_epoch()
        if self._use_ddp:
            dist.all_reduce(self._round_loss_sums, op=dist.ReduceOp.SUM)
        (first, second, count, roi_count) = self._round_loss_sums.tolist()
        self._epoch_round_losses = (first / count, second / count) if count else (None, None)
        self._train_roi_per_image = roi_count / count if count else None
        if _teacher_forcing_options(self.cfg)['enabled']:
            record = {'epoch': self.current_epoch + 1, 'proposal_source': phase,
                      'roi_count_per_image': self._train_roi_per_image}
            if not hasattr(self, '_teacher_forcing_history'):
                self._teacher_forcing_history = []
            self._teacher_forcing_history.append(record)
        mean_loss = (first + float(self._network_module.loss_weight) * second) / max(count, 1)
        return (mean_loss, timing)

    def pop_epoch_round_losses(self):
        """Reuse the existing paired-loss logger columns for the two rounds."""
        result = getattr(self, '_epoch_round_losses', (None, None))
        self._epoch_round_losses = (None, None)
        return result

    def _forward_eval_batch(self, model, images, batch):
        module = model.module if hasattr(model, 'module') else model
        probabilities = module.predict(images, diagnostics=True, return_base_canvas=False)
        return probabilities.clamp_min(1e-07).log()

    @torch.no_grad()
    def _run_eval(self, loader, compute_hd, distributed=False):
        """Overlap bounded CPU metrics with GPU prediction; gather scalar records."""
        from rescale.evaluation.validation import MetricQueue, aggregate_records
        self.network.eval()
        if self._use_ddp and (not distributed) and (not self.is_main):
            return self._broadcast_from_main(None)
        module = self._network_module
        if self._use_ddp and distributed:
            for buffer in module.buffers():
                dist.broadcast(buffer, src=0)
        eval_loss = getattr(self.loss_fn, 'base_loss', self.loss_fn)
        if not isinstance(eval_loss, DiceCELoss):
            eval_loss = DiceCELoss(ce_weight=float(self.cfg['training'].get('loss_ce_weight', 0.5)))
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        inter = torch.zeros(self.num_classes, device=self.device, dtype=torch.float64)
        union = torch.zeros_like(inter)
        roi_total = torch.zeros_like(loss_sum)
        (count, batch_count) = (0, 0)
        recursive = bool(module.recursion_enabled) and int(module.rounds) == 2
        workers = self.cfg.get('metrics', {}).get('val_workers', 2)
        requested = self.cfg.get('metrics', {}).get('val')
        include_recall = (requested is None or 'recall' in requested) and getattr(self, '_compute_recall', True)
        label_dtype = torch.uint8 if self.num_classes <= 256 else torch.int32
        with MetricQueue(workers) as queue:
            for batch in loader:
                images = batch['image'].to(self.device, non_blocking=True)
                with self._autocast_context():
                    logits = self._forward_eval_batch(module, images, batch)
                targets = batch['mask'].to(self.device, non_blocking=True).long()
                self._legacy_eval_size = tuple(targets.shape[-2:])
                sizes = [targets.shape[-2:]] * images.shape[0]
                aux = module.get_aux_outputs()
                base_logits = aux.get('base_input_logits') if recursive else logits
                coverage = aux.get('coverage') if recursive else None
                if recursive:
                    roi_counts = aux.get('roi_counts')
                    if roi_counts is None or roi_counts.shape != (images.shape[0],):
                        raise ValueError('Recursive prediction diagnostics require per-image roi_counts')
                    roi_total += roi_counts.sum().double()
                predictions = logits.argmax(1).to(label_dtype)
                if recursive:
                    base_predictions = base_logits.argmax(1).to(label_dtype) if base_logits is not None else None
                else:
                    base_predictions = predictions
                loss_sum += eval_loss(logits, targets).double()
                batch_count += 1
                for cls in range(1, self.num_classes):
                    (pred_c, target_c) = (predictions == cls, targets == cls)
                    inter[cls] += (pred_c & target_c).sum()
                    union[cls] += pred_c.sum() + target_c.sum()
                cpu_predictions = predictions.cpu().numpy()
                cpu_targets = targets.to(label_dtype).cpu().numpy()
                if recursive:
                    cpu_base = base_predictions.cpu().numpy() if base_predictions is not None else None
                else:
                    cpu_base = cpu_predictions
                cpu_coverage = coverage.bool().cpu().numpy() if coverage is not None else None
                for (index, (height, width)) in enumerate(sizes):
                    prediction = cpu_predictions[index]
                    if recursive:
                        base = cpu_base[index] if cpu_base is not None else None
                    else:
                        base = prediction
                    (covered, coverage_target) = (None, None)
                    if cpu_coverage is not None:
                        covered = cpu_coverage[index, 0, :height, :width] if cpu_coverage.ndim == 4 else cpu_coverage[index, :height, :width]
                        coverage_target = cpu_targets[index, :height, :width]
                    queue.submit(prediction, cpu_targets[index], base, covered, self.num_classes, compute_hd, recursive, coverage_target, include_recall)
                    count += 1
                    del prediction, base, covered, coverage_target
                module.clear_aux_outputs()
                del images, logits, targets, predictions, base_predictions, aux, base_logits, coverage, batch
                del cpu_predictions, cpu_targets, cpu_base, cpu_coverage, pred_c, target_c
                if recursive:
                    del roi_counts
            records = queue.finish()
        packed = torch.cat((torch.stack((loss_sum, loss_sum.new_tensor(batch_count), loss_sum.new_tensor(count), roi_total)), inter, union))
        if self._use_ddp and distributed:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, records)
            if self.is_main:
                records = [record for shard in gathered for record in shard]
        (loss_sum, batch_count) = (packed[0], int(packed[1].item()))
        (count, roi_total) = (int(packed[2].item()), packed[3])
        (inter, union) = (packed[4:4 + self.num_classes], packed[4 + self.num_classes:])
        (exact, extra, diagnostics) = ({}, {}, {})
        if self.is_main:
            (exact, extra, diagnostics) = aggregate_records(records, compute_hd, recursive)
        pseudo = ((2 * inter[1:] + 1e-05) / (union[1:] + 1e-05)).mean().item()
        result = self._broadcast_from_main({'loss': loss_sum.item() / max(batch_count, 1), 'pseudo_dice': pseudo, 'dice': exact.get('dice'), 'iou': exact.get('iou'), 'hd95': exact.get('hd95'), 'extra_metrics': extra, 'roi_count_per_image': roi_total.item() / max(count, 1) if recursive else None, 'diagnostics': diagnostics})
        self._last_base_hd95 = result['extra_metrics'].get('base', {}).get('hd95')
        self._last_diagnostics = result['diagnostics']
        return result


    def validate_one_epoch(self, compute_hd=False, *, full_evaluation=False):
        interval = self._val_hd95_interval
        last_epoch = self.current_epoch + 1 == getattr(self, 'num_epochs', 0)
        self._compute_recall = full_evaluation or compute_hd or last_epoch or (self.current_epoch % interval == 0)
        result = self._validate_epoch(compute_hd=compute_hd)
        base_hd95 = self._last_base_hd95 if compute_hd else None
        if compute_hd and 'hd95' in self._val_metrics:
            result['base_hd95'] = base_hd95
        result.update(getattr(self, '_last_diagnostics', {}))
        return result

    def validate_checkpoint(self, compute_hd=False):
        return self.validate_one_epoch(compute_hd=compute_hd, full_evaluation=True)

    def save_checkpoint(self, path):
        if self.is_main:
            torch.save({'model_state': self._network_module.state_dict(), 'config': self.cfg, 'epoch': self.current_epoch, 'recursive_seg_version': 2, 'evaluation_protocol': self._evaluation_protocol()}, path)

    def _evaluation_protocol(self):
        size = self.cfg.get('data', {}).get('image_size', getattr(self, '_legacy_eval_size', None))
        return {'name': 'legacy_resized_v1', 'image_size': list(size) if size is not None else None, 'prediction': 'argmax', 'target': 'existing_validation_transform_mask', 'refined_resampling': 'bilinear_probabilities_align_corners_false', 'hd95_unit': 'resized_pixels', 'empty_masks': 'existing_nan_skip'}

    def run(self):
        """Record the complete fit, including validation/logging/checkpoint I/O."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        self._fit()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        duration = time.perf_counter() - started
        if self.is_main:
            path = Path(self.output_dir) / 'results.json'
            with path.open('r', encoding='utf-8') as stream:
                result = json.load(stream)
            module = self._network_module
            backbone_parameters = sum((parameter.numel() for parameter in module.backbone.parameters()))
            result['runtime'] = {'fit_wall_time_seconds': duration, 'fit_scope': 'Includes epochs, validation, checkpoint I/O, logging and final evaluation; excludes trainer/data-cache construction', 'device': str(self.device), 'amp_dtype': str(getattr(self, 'amp_dtype', torch.float16)) if self.amp_enabled else None, 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None, 'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device) if self.device.type == 'cuda' else None, 'parameter_count': sum((parameter.numel() for parameter in module.parameters())), 'shared_backbone_parameter_count': backbone_parameters, 'crop_source': getattr(module, 'crop_source', 'native'), 'sampling_strategy': getattr(module, 'sampling_strategy', 'crop'), 'outside_roi_background': module.rounds == 2 and module.outside_roi_background}
            if getattr(self, '_shared_cache_report', None) is not None:
                result['shared_cache_preparation'] = self._shared_cache_report
            if _teacher_forcing_options(self.cfg)['enabled']:
                result['teacher_forcing_training'] = {'epochs': getattr(self, '_teacher_forcing_history', [])}
            with path.open('w', encoding='utf-8') as stream:
                json.dump(_finite_json(result), stream, indent=2, allow_nan=False)

    def _write_results(self, final_metrics=None):
        self._save_result_summary(final_metrics=final_metrics)
        path = Path(self.output_dir) / 'results.json'
        with path.open('r', encoding='utf-8') as stream:
            result = json.load(stream)
        additional = {key: value for (key, value) in (final_metrics or {}).items() if key not in {'dice', 'iou', 'hd95', 'pseudo_dice', 'val_loss'}}
        if additional:
            result.setdefault('best_checkpoint_eval', {}).update(additional)
        for key in ('dice', 'iou', 'hd95'):
            if key in (final_metrics or {}):
                result['val'][key] = final_metrics[key]
        result['evaluation_protocol'] = self._evaluation_protocol()
        result['innovations'] = self.cfg.get('innovations', {})
        module = self._network_module
        if hasattr(module, 'refinement_metadata'):
            result['refinement_strategy'] = module.refinement_metadata()
        result['recursive_backbone'] = {'requested': getattr(module, 'backbone_name', 'RecursiveUNet'), 'resolved': getattr(module, 'resolved_backbone_name', 'RecursiveUNet'), 'uses_scale': getattr(module, 'uses_scale', True)}
        with path.open('w', encoding='utf-8') as stream:
            json.dump(_finite_json(result), stream, indent=2, allow_nan=False)
