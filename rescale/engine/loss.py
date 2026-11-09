from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation.

    Args:
        smooth: Smoothing constant to avoid division by zero.
        apply_softmax: If True, apply softmax to logits before computing dice.
    """

    def __init__(self, smooth: float=1e-05, apply_softmax: bool=True):
        super().__init__()
        self.smooth = smooth
        self.apply_softmax = apply_softmax

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute mean foreground soft Dice loss.

        Args:
            logits: (B, C, H, W) raw network output.
            targets: (B, H, W) integer class labels.

        Returns:
            Scalar loss averaged over foreground classes and batch.
        """
        logits = logits.float()
        num_classes = logits.shape[1]
        if self.apply_softmax:
            probs = torch.softmax(logits, dim=1)
        else:
            probs = logits
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        probs_flat = probs.view(probs.shape[0], num_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_fg = dice_per_class[:, 1:].mean()
        return 1.0 - dice_fg

class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss.

    Args:
        ce_weight: Weight for cross-entropy term. Dice weight = 1 - ce_weight.
        smooth: Smoothing for soft Dice.
        ignore_index: Ignored label index passed to CrossEntropyLoss.
    """

    def __init__(self, ce_weight: float=0.5, smooth: float=1e-05, ignore_index: int=-100):
        super().__init__()
        self.dice_weight = 1.0 - ce_weight
        self.ce_weight = ce_weight
        self.dice = SoftDiceLoss(smooth=smooth)
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice + CE loss.

        Args:
            logits: (B, C, H, W) raw network output.
            targets: (B, H, W) integer class labels.

        Returns:
            Scalar combined loss.
        """
        logits = logits.float()
        return self.dice_weight * self.dice(logits, targets) + self.ce_weight * self.ce(logits, targets)

class DeepSupervisionLoss(nn.Module):
    """Wraps a base loss to handle deep supervision outputs.

    Computes the weighted sum of losses at multiple scales. The first element
    in the output list is assumed to be the highest-resolution prediction.

    Args:
        base_loss: Any segmentation loss callable.
        weights: Per-scale loss weights (should sum to 1 or be normalized).
            Length must match the number of deep-supervision outputs.
    """

    def __init__(self, base_loss: nn.Module, weights: List[float]):
        super().__init__()
        self.base_loss = base_loss
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def forward(self, outputs: Union[torch.Tensor, List[torch.Tensor]], targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted multi-scale loss.

        Args:
            outputs: List of (B, C, H_s, W_s) tensors from high to low res,
                or a single tensor when deep supervision is disabled.
            targets: (B, H, W) ground-truth labels at the original resolution.

        Returns:
            Weighted loss scalar.
        """
        if not isinstance(outputs, (list, tuple)):
            return self.base_loss(outputs, targets)
        loss = 0.0
        for (i, pred) in enumerate(outputs):
            w = self.weights[i] if i < len(self.weights) else self.weights[-1]
            if pred.shape[-2:] != targets.shape[-2:]:
                tgt = _resize_targets(targets, pred.shape[-2:])
            else:
                tgt = targets
            loss = loss + w * self.base_loss(pred, tgt)
        return loss

def _resize_targets(targets: torch.Tensor, size: tuple) -> torch.Tensor:
    """Nearest-neighbour resize integer label map.

    Args:
        targets: (B, H, W) long tensor.
        size: Target (H', W').

    Returns:
        (B, H', W') long tensor.
    """
    return F.interpolate(targets.unsqueeze(1).float(), size=size, mode='nearest').squeeze(1).long()

def build_loss(cfg: dict) -> nn.Module:
    """Build the training loss from config.

    Args:
        cfg: Full config dict. Reads model.deep_supervision / aux_supervision,
            model.ds_weights / aux_weights, and training.loss_ce_weight.

    Returns:
        A loss module that matches the configured supervision strategy.
    """
    ce_weight = cfg['training'].get('loss_ce_weight', 0.5)
    base_loss = DiceCELoss(ce_weight=ce_weight)
    if cfg['model'].get('deep_supervision', False):
        weights = cfg['model'].get('ds_weights', [1.0, 0.5, 0.25, 0.125])
        return DeepSupervisionLoss(base_loss, weights)
    return base_loss
