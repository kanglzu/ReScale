"""Train ReScale on one existing fold; torchrun enables distributed training."""
import argparse
import os
from rescale.config import DATASETS, load_config, apply_overrides

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/rescale.yaml')
    p.add_argument('--dataset', choices=DATASETS, help='Override the dataset in the shared config')
    p.add_argument('--fold', type=int, choices=range(5), default=0)
    p.add_argument('--stop-after-epochs', type=int, default=None)
    p.add_argument('--cfg-options', nargs='*', default=[])
    a = p.parse_args()
    cfg = apply_overrides(load_config(a.config, dataset=a.dataset), a.cfg_options)
    cfg['training']['fold'] = a.fold
    if a.stop_after_epochs is not None: cfg['training']['stop_after_epochs'] = a.stop_after_epochs
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    from rescale.engine.trainer import ReScaleTrainer
    ReScaleTrainer(cfg).run()

if __name__ == '__main__': main()
