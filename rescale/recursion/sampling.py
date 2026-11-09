"""Stateless, training-only uniform sampling of one lesion per image."""

import hashlib
import json
import random

import torch


def one_roi_training_enabled(cfg):
    value = cfg.get("innovations", {}).get("recursion", {}).get("train_one_roi_per_image", False)
    if not isinstance(value, bool):
        raise ValueError("recursion.train_one_roi_per_image must be true or false")
    return value


def _uniform_draw(sample_id, seed, fold, epoch):
    # Do not use Python hash() or consume the model/augmentation RNG streams.
    key = json.dumps([int(seed), int(fold), int(epoch), str(sample_id)], separators=(",", ":"))
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
    return random.Random(int.from_bytes(digest, "little")).random()


@torch.no_grad()
def select_one_roi_per_image(boxes, batch_indices, sample_ids, *, seed, fold, epoch):
    """Return B row indices into the original ROI tensors, in image order.

    Candidate boxes are sorted by image and coordinates before sampling, so
    proposal order and DDP rank do not control the chosen lesion. Empty-mask
    whole-image fallbacks must already have been supplied by the proposal code.
    """
    batch = len(sample_ids)
    if (batch < 1 or boxes.ndim != 2 or boxes.shape[1] != 4
            or batch_indices.shape != (len(boxes),) or batch_indices.dtype != torch.long
            or batch_indices.device != boxes.device or len(boxes) < batch):
        raise ValueError("One-ROI training requires boxes, image indices and one sample ID per image")
    counts = torch.zeros(batch, device=boxes.device, dtype=torch.long)
    counts.scatter_add_(0, batch_indices, torch.ones_like(batch_indices))
    # Asynchronous on CUDA: do not download counts or box coordinates to Python.
    torch._assert_async((counts > 0).all(), "Every image needs a proposal or whole-image fallback")
    if len(boxes) == batch:
        return torch.argsort(batch_indices)

    order = torch.arange(len(boxes), device=boxes.device)
    for coordinate in (3, 2, 1, 0):
        order = order[torch.argsort(boxes[order, coordinate], stable=True)]
    order = order[torch.argsort(batch_indices[order], stable=True)]
    draws = [_uniform_draw(sid, seed, fold, epoch) for sid in sample_ids]
    uniform = torch.tensor(draws, device=boxes.device, dtype=torch.float64)
    offsets = counts.cumsum(0) - counts
    local_indices = (uniform * counts).long()
    return order[offsets + local_indices]
