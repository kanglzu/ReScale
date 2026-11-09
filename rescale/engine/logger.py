"""
Training logger with real-time visualization.

Writes a structured training_log.txt (nnUNet-style) and updates a
progress.png after every epoch.

Log format (per epoch):
    [Epoch 010/300] train_loss=0.4321  val_loss=0.3812  pseudo_dice=0.7341
                    ema_pseudo_dice=0.7156  dice=0.8012  iou=0.6923  hd95=12.34
                    lr=9.5e-05  epoch=42.1s  train=33.9s  val=8.2s  [BEST]
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingLogger:
    """Logs metrics to a text file and renders a progress plot.

    Args:
        output_dir: Directory where log files are saved.
        num_epochs: Total planned epochs (for axis scaling).
    """
    _METRIC_KEYS = ['train_loss', 'global_loss', 'focused_loss', 'val_loss', 'pseudo_dice', 'ema_pseudo_dice', 'dice', 'iou', 'base_dice', 'base_iou', 'roi_count_per_image', 'train_roi_per_image', 'hd95', 'lr', 'epoch_time', 'time_train', 'time_val']

    def __init__(self, output_dir: str, num_epochs: int):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.log_file = self.output_dir / 'training_log.txt'
        self.state_file = self.output_dir / 'logger_state.json'
        self._data: Dict[str, List] = {k: [] for k in self._METRIC_KEYS}
        self._best_epoch: Optional[int] = None
        self._best_metric_value: Optional[float] = None

    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float], pseudo_dice: Optional[float], ema_pseudo_dice: Optional[float], dice: Optional[float], iou: Optional[float], lr: float, epoch_time: float, hd95: Optional[float]=None, is_best: bool=False, timing: Optional[Dict[str, float]]=None, global_loss: Optional[float]=None, focused_loss: Optional[float]=None, base_dice: Optional[float]=None, base_iou: Optional[float]=None, roi_count_per_image: Optional[float]=None, train_roi_per_image: Optional[float]=None) -> None:
        """Record one epoch of metrics and update all outputs.

        Args:
            epoch: 0-based epoch index.
            train_loss: Mean training loss for this epoch.
            val_loss: Mean validation loss for this epoch.
            pseudo_dice: Pseudo-Dice computed on the validation set (batch-level).
            ema_pseudo_dice: EMA-smoothed pseudo-Dice.
            dice: Exact mean foreground Dice on the validation set.
            iou: Exact mean foreground IoU on the validation set.
            lr: Current learning rate.
            epoch_time: Total wall-clock time for this epoch in seconds.
            hd95: HD95 value (None when not computed this epoch).
            is_best: Whether this epoch achieved the trainer's selection metric.
            timing: Optional timing stats for this epoch, including train/val
                phase wall-clock time.
        """
        self._data['train_loss'].append(train_loss)
        self._data['global_loss'].append(global_loss)
        self._data['focused_loss'].append(focused_loss)
        self._data['val_loss'].append(val_loss)
        self._data['pseudo_dice'].append(pseudo_dice)
        self._data['ema_pseudo_dice'].append(ema_pseudo_dice)
        self._data['dice'].append(dice)
        self._data['iou'].append(iou)
        self._data['base_dice'].append(base_dice)
        self._data['base_iou'].append(base_iou)
        self._data['roi_count_per_image'].append(roi_count_per_image)
        self._data['train_roi_per_image'].append(train_roi_per_image)
        self._data['hd95'].append(hd95)
        self._data['lr'].append(lr)
        self._data['epoch_time'].append(epoch_time)
        timing = timing or {}
        self._data['time_train'].append(timing.get('train'))
        self._data['time_val'].append(timing.get('val'))
        if is_best:
            self._best_epoch = epoch
            self._best_metric_value = dice
        self._write_log_line(epoch, is_best)
        self._save_state()
        self._plot_progress()

    def print_epoch_summary(self, epoch: int) -> None:
        """Print the latest epoch summary to stdout."""
        i = -1
        parts = [f'[Epoch {epoch + 1:04d}/{self.num_epochs:04d}]', f"  train_loss={self._data['train_loss'][i]:.4f}"]
        if self._has_round_losses():
            if self._data['global_loss'][i] is not None:
                parts.append(f"  base_loss={self._data['global_loss'][i]:.4f}")
            if self._data['focused_loss'][i] is not None:
                parts.append(f"  polish_loss={self._data['focused_loss'][i]:.4f}")

        def _fmt(v: Optional[float], spec: str) -> str:
            return f'{v:{spec}}' if v is not None else '--'
        parts.extend([f"  val_loss={_fmt(self._data['val_loss'][i], '.4f')}", f"  pseudo_dice={_fmt(self._data['pseudo_dice'][i], '.4f')}", f"  ema_pseudo_dice={_fmt(self._data['ema_pseudo_dice'][i], '.4f')}"])
        if self._data['dice'][i] is not None:
            parts.append(f"  dice={self._data['dice'][i]:.4f}")
        if self._data['iou'][i] is not None:
            parts.append(f"  iou={self._data['iou'][i]:.4f}")
        if self._data['base_dice'][i] is not None:
            parts.append(f"  base_dice={self._data['base_dice'][i]:.4f}")
        if self._data['base_iou'][i] is not None:
            parts.append(f"  base_iou={self._data['base_iou'][i]:.4f}")
        if self._data['roi_count_per_image'][i] is not None:
            parts.append(f"  roi_per_image={self._data['roi_count_per_image'][i]:.2f}")
        if self._data['train_roi_per_image'][i] is not None:
            parts.append(f"  train_roi_per_image={self._data['train_roi_per_image'][i]:.2f}")
        if self._data['hd95'][i] is not None:
            parts.append(f"  hd95={self._data['hd95'][i]:.2f}")
        parts.append(f"  lr={self._data['lr'][i]:.3e}")
        parts.append(f"  epoch={self._data['epoch_time'][i]:.1f}s")
        parts.extend(self._format_timing_parts(i))
        if epoch == self._best_epoch:
            parts.append('  [BEST]')
        print(''.join(parts))

    def _has_round_losses(self) -> bool:
        """Return whether global or focused losses have ever been recorded."""
        return any((v is not None for v in self._data['global_loss'])) or any((v is not None for v in self._data['focused_loss']))

    def _format_timing_parts(self, index: int) -> List[str]:
        """Format timing stats for stdout and text logs."""
        parts = []
        train_time = self._data['time_train'][index]
        if train_time is not None:
            parts.append(f'  train={train_time:.1f}s')
        val_time = self._data['time_val'][index]
        if val_time is not None:
            parts.append(f'  val={val_time:.1f}s')
        return parts

    def _write_log_line(self, epoch: int, is_best: bool) -> None:
        i = -1
        parts = [f'[Epoch {epoch + 1:04d}/{self.num_epochs:04d}]', f"  train_loss={self._data['train_loss'][i]:.4f}"]
        if self._has_round_losses():
            if self._data['global_loss'][i] is not None:
                parts.append(f"  base_loss={self._data['global_loss'][i]:.4f}")
            if self._data['focused_loss'][i] is not None:
                parts.append(f"  polish_loss={self._data['focused_loss'][i]:.4f}")
        for name in ('val_loss', 'pseudo_dice', 'ema_pseudo_dice'):
            value = self._data[name][i]
            parts.append(f'  {name}={value:.4f}' if value is not None else f'  {name}=--')
        if self._data['dice'][i] is not None:
            parts.append(f"  dice={self._data['dice'][i]:.4f}")
        if self._data['iou'][i] is not None:
            parts.append(f"  iou={self._data['iou'][i]:.4f}")
        if self._data['base_dice'][i] is not None:
            parts.append(f"  base_dice={self._data['base_dice'][i]:.4f}")
        if self._data['base_iou'][i] is not None:
            parts.append(f"  base_iou={self._data['base_iou'][i]:.4f}")
        if self._data['roi_count_per_image'][i] is not None:
            parts.append(f"  roi_per_image={self._data['roi_count_per_image'][i]:.2f}")
        if self._data['train_roi_per_image'][i] is not None:
            parts.append(f"  train_roi_per_image={self._data['train_roi_per_image'][i]:.2f}")
        if self._data['hd95'][i] is not None:
            parts.append(f"  hd95={self._data['hd95'][i]:.2f}")
        parts.append(f"  lr={self._data['lr'][i]:.3e}")
        parts.append(f"  epoch={self._data['epoch_time'][i]:.1f}s")
        parts.extend(self._format_timing_parts(i))
        if is_best:
            parts.append('  [BEST]')
        with open(self.log_file, 'a') as f:
            f.write(''.join(parts) + '\n')

    def _save_state(self) -> None:
        state = {'data': self._data, 'best_epoch': self._best_epoch, 'best_metric_value': self._best_metric_value}
        with open(self.state_file, 'w') as f:
            json.dump(state, f)

    def _plot_progress(self) -> None:
        """Render and save progress.png with 6 sub-plots (3 rows × 2 cols).

        Layout:
          Row 1: Train/Val Loss  |  Learning Rate
          Row 2: pDice + EMA     |  Exact Dice
          Row 3: Exact IoU       |  HD95
        """
        epochs = list(range(1, len(self._data['train_loss']) + 1))
        if not epochs:
            return
        sns.set_theme(font_scale=1.2)
        (fig, axes) = plt.subplots(3, 2, figsize=(18, 18))
        best_vline_kw = dict(color='green', linestyle='--', alpha=0.5, linewidth=1.5)

        def _vline(ax):
            if self._best_epoch is not None:
                ax.axvline(self._best_epoch + 1, **best_vline_kw)
        ax = axes[0, 0]
        ax.plot(epochs, self._data['train_loss'], color='steelblue', linewidth=2, label='train')
        ax.plot(epochs, self._data['val_loss'], color='tomato', linewidth=2, label='val')
        if self._has_round_losses():
            base_vals = [v for v in self._data['global_loss'] if v is not None]
            base_epochs = [e for (e, v) in zip(epochs, self._data['global_loss']) if v is not None]
            refined_vals = [v for v in self._data['focused_loss'] if v is not None]
            refined_epochs = [e for (e, v) in zip(epochs, self._data['focused_loss']) if v is not None]
            if base_vals:
                ax.plot(base_epochs, base_vals, color='slategray', linewidth=1.8, linestyle='--', label='base')
            if refined_vals:
                ax.plot(refined_epochs, refined_vals, color='darkgreen', linewidth=1.8, linestyle='-.', label='polish')
        ax.set_title('Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        _vline(ax)
        ax = axes[0, 1]
        ax.plot(epochs, self._data['lr'], color='darkorange', linewidth=2)
        ax.set_title('Learning Rate')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('LR')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax = axes[1, 0]
        ax.plot(epochs, self._data['pseudo_dice'], color='mediumseagreen', linewidth=1.5, linestyle='--', label='Pseudo Dice')
        ax.plot(epochs, self._data['ema_pseudo_dice'], color='seagreen', linewidth=2.5, label='EMA Pseudo Dice')
        ax.set_title('Pseudo Dice / EMA Pseudo Dice')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Pseudo Dice')
        ax.legend()
        _vline(ax)
        ax = axes[1, 1]
        dice_vals = [v for v in self._data['dice'] if v is not None]
        dice_epochs = [e for (e, v) in zip(epochs, self._data['dice']) if v is not None]
        if dice_vals:
            ax.plot(dice_epochs, dice_vals, color='royalblue', linewidth=2, label='Dice')
        base_pairs = [(e, v) for (e, v) in zip(epochs, self._data['base_dice']) if v is not None]
        if base_pairs:
            (base_epochs, base_values) = zip(*base_pairs)
            ax.plot(base_epochs, base_values, color='slategray', linewidth=1.5, linestyle='--', label='Base Dice')
        ax.set_title('Exact Dice')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice')
        ax.legend()
        _vline(ax)
        ax = axes[2, 0]
        iou_vals = [v for v in self._data['iou'] if v is not None]
        iou_epochs = [e for (e, v) in zip(epochs, self._data['iou']) if v is not None]
        if iou_vals:
            ax.plot(iou_epochs, iou_vals, color='mediumpurple', linewidth=2, label='IoU')
        ax.set_title('Exact IoU')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('IoU')
        ax.legend()
        _vline(ax)
        ax = axes[2, 1]
        hd_pairs = [(e, v) for (e, v) in zip(epochs, self._data['hd95']) if v is not None]
        if hd_pairs:
            (hd_epochs, hd_vals) = zip(*hd_pairs)
            ax.plot(hd_epochs, hd_vals, color='darksalmon', linewidth=2, label='HD95')
            ax.set_ylabel('HD95 (px)')
        else:
            ax.text(0.5, 0.5, 'HD95 not enabled', transform=ax.transAxes, ha='center', va='center', color='gray')
        ax.set_title('HD95')
        ax.set_xlabel('Epoch')
        ax.legend()
        _vline(ax)
        plt.tight_layout()
        fig.savefig(self.output_dir / 'progress.png', dpi=100)
        plt.close(fig)
