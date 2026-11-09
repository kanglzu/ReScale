"""Per-image segmentation metrics and the validation EMA tracker."""
from typing import Optional

import numpy as np


def compute_recall(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    """Mean foreground recall, skipping classes absent from the target."""
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.ndim != 2 or pred.shape != target.shape:
        raise ValueError("Prediction and target must have matching H,W")
    values = []
    for cls in range(1, num_classes):
        foreground = target == cls
        area = int(foreground.sum())
        if area:
            values.append(float(((pred == cls) & foreground).sum() / area))
    return float(np.mean(values)) if values else float("nan")


def compute_dice(pred: np.ndarray, target: np.ndarray, num_classes: int, smooth: float = 1e-5) -> float:
    """Compute mean foreground Dice coefficient for a single sample.

    Args:
        pred: (H, W) predicted class map.
        target: (H, W) ground-truth class map.
        num_classes: Total number of classes (including background).
        smooth: Smoothing constant.

    Returns:
        Mean Dice over foreground classes.
    """
    dice_values = []
    for c in range(1, num_classes):
        pred_c = (pred == c)
        tgt_c  = (target == c)
        inter  = (pred_c & tgt_c).sum()
        union  = pred_c.sum() + tgt_c.sum()
        if union == 0:
            # Both prediction and target have no pixels of this class; skip so
            # validation averages only defined scores.
            dice_values.append(float("nan"))
        else:
            dice_values.append((2.0 * inter + smooth) / (union + smooth))
    valid = [v for v in dice_values if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int, smooth: float = 1e-5) -> float:
    """Compute mean foreground IoU (Jaccard) for a single sample.

    Args:
        pred: (H, W) predicted class map.
        target: (H, W) ground-truth class map.
        num_classes: Total number of classes (including background).
        smooth: Smoothing constant.

    Returns:
        Mean IoU over foreground classes.
    """
    iou_values = []
    for c in range(1, num_classes):
        pred_c = (pred == c)
        tgt_c  = (target == c)
        inter  = (pred_c & tgt_c).sum()
        union  = (pred_c | tgt_c).sum()
        if union == 0:
            iou_values.append(float("nan"))
        else:
            iou_values.append((inter + smooth) / (union + smooth))
    valid = [v for v in iou_values if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_hd95(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    """Compute mean foreground HD95 (95th-percentile Hausdorff distance).

    Uses medpy for surface distance computation.  Falls back to NaN if either
    the prediction or the target mask is empty for a given class.

    Args:
        pred: (H, W) predicted class map.
        target: (H, W) ground-truth class map.
        num_classes: Total number of classes (including background).

    Returns:
        Mean HD95 over foreground classes (mm equivalent for 2D unit spacing).
        Returns NaN when no foreground exists in either pred or target.
    """
    try:
        from medpy.metric.binary import hd95 as medpy_hd95
    except ImportError:
        raise ImportError(
            "medpy is required for HD95 computation. "
            "Install it with: pip install medpy"
        )

    hd95_values = []
    for c in range(1, num_classes):
        pred_c = (pred == c)
        tgt_c  = (target == c)
        if pred_c.sum() == 0 or tgt_c.sum() == 0:
            hd95_values.append(float("nan"))
        else:
            hd95_values.append(medpy_hd95(pred_c, tgt_c))

    valid = [v for v in hd95_values if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


class EMATracker:
    """Exponential moving average tracker for a scalar metric.

    Args:
        alpha: Smoothing coefficient in [0, 1). Higher = more weight on history.
    """

    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self._value: Optional[float] = None

    def update(self, value: float) -> float:
        """Update EMA with a new observation.

        Args:
            value: New metric value.

        Returns:
            Current EMA value after update.
        """
        if self._value is None:
            self._value = value
        else:
            self._value = self.alpha * self._value + (1.0 - self.alpha) * value
        return self._value

    @property
    def value(self) -> Optional[float]:
        """Current EMA value (None if no update has been called yet)."""
        return self._value
