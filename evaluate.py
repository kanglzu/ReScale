"""Evaluate a trained checkpoint with the original all-ROI validation protocol."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
import argparse
import json
from pathlib import Path
import torch
from rescale.config import DATASETS, load_config, apply_overrides
from rescale.engine.trainer import ReScaleTrainer, _finite_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/rescale.yaml')
    p.add_argument('--dataset', choices=DATASETS, help='Override the dataset in the shared config')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--fold', type=int, choices=range(5), required=True)
    p.add_argument('--output', type=Path, default=Path('outputs/evaluation'))
    p.add_argument('--cfg-options', nargs='*', default=[])
    a = p.parse_args()
    cfg = apply_overrides(load_config(a.config, dataset=a.dataset), a.cfg_options)
    cfg['training']['fold'] = a.fold
    cfg['output'].update(save_dir=str(a.output), experiment_name='evaluation')
    trainer = ReScaleTrainer(cfg)
    ckpt = torch.load(a.checkpoint, map_location=trainer.device)
    trainer._network_module.load_state_dict(trainer._checkpoint_model_state(ckpt), strict=True)
    trainer.current_epoch = int(ckpt.get('epoch', 199))
    metrics = trainer.validate_checkpoint(compute_hd=True)
    if trainer.is_main:
        output = trainer.output_dir/'evaluation.json'
        output.write_text(json.dumps(_finite_json(metrics), indent=2, allow_nan=False))
        print(json.dumps(_finite_json(metrics), indent=2, allow_nan=False))
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
