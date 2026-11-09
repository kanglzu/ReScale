"""Lesion proposals, indexed crops and probability merging.

Boxes are continuous XYXY edges in the first-round image coordinate system.
Pixel centers have coordinates (x + .5, y + .5). Sampling and probability paste
use align_corners=False. Hard proposals, raw images and paste need no gradients;
the second-round segmentation loss is computed directly on the sampled target.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def _check_logits(logits: Tensor) -> None:
    if not isinstance(logits, Tensor) or logits.ndim != 4 or min(logits.shape) <= 0:
        raise ValueError("Expected nonempty BCHW logits")
    if logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Logits must be FP32, FP16 or BF16")


def _check_backend(backend: str) -> None:
    if backend not in {"reference", "cuda"}:
        raise ValueError(f"Unknown recursive geometry backend: {backend}")


def _size(value, name: str) -> Tuple[int, int]:
    if len(value) != 2 or any(not isinstance(v, int) or v <= 0 for v in value):
        raise ValueError(f"{name} must contain two positive integer spatial dimensions")
    return tuple(value)


def _check_boxes(boxes: Tensor, batch: int, device) -> None:
    if boxes.ndim != 2 or boxes.shape != (batch, 4):
        raise ValueError("One XYXY box per image is required: [B,4]")
    if boxes.device != device or not boxes.is_floating_point():
        raise ValueError("Boxes must be floating point on the image device")


@torch.no_grad()
def lesion_boxes_from_logits(logits, min_area=1, margin=.1, threshold=.5,
                             backend="reference"):
    """Separate class-aware 8-connected components; empty images use one full box.

    min_area counts foreground pixels before expansion on the model input grid.
    The reference backend uses CPU connected components. The CUDA backend keeps
    component labels, boxes and source indices on GPU; selecting a variable
    number of components requires a dynamic-size synchronization.
    """
    _check_logits(logits)
    if not 0 < threshold < 1:
        raise ValueError("Invalid lesion margin or threshold")
    values = logits.detach().float()
    cutoff = math.log(threshold / (1 - threshold))
    if values.shape[1] == 1:
        labels = (values[:, 0] >= cutoff).int()
    elif values.shape[1] == 2:
        labels = ((values[:, 1] - values[:, 0]) >= cutoff).int()
    else:
        labels = values.argmax(1).int()
    return lesion_boxes_from_labels(labels, min_area=min_area, margin=margin, backend=backend)


@torch.no_grad()
def lesion_boxes_from_labels(labels, min_area=1, margin=.1, backend="reference"):
    """Per-lesion boxes directly from categorical GT, without one-hot logits."""
    _check_backend(backend)
    integral_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32,
                       torch.int64, torch.bool)
    if (not isinstance(labels, Tensor) or labels.ndim != 3 or min(labels.shape) < 1
            or labels.dtype not in integral_dtypes):
        raise ValueError("Lesion labels must be nonempty integral [B,H,W]")
    if isinstance(min_area, bool) or not isinstance(min_area, int) or min_area < 1:
        raise ValueError("min_area must be a positive integer")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Invalid lesion margin")
    labels = labels.detach().to(torch.int32)
    batch, height, width = labels.shape
    if backend == "cuda":
        from rescale.ops import cuda_components
        boxes, indices = cuda_components(labels.contiguous(), min_area, margin)
        counts = torch.zeros(batch, device=indices.device, dtype=torch.int64)
        counts.scatter_add_(0, indices, torch.ones_like(indices))
        missing = torch.nonzero(counts == 0).flatten()
        full = boxes.new_tensor((0., 0., float(width), float(height)))
        boxes = torch.cat((boxes, full[None].expand(missing.numel(), 4)))
        indices = torch.cat((indices, missing))
        order = indices.argsort(stable=True)
        return boxes[order], indices[order]

    import cv2
    import numpy as np
    labels_cpu = labels.cpu().numpy()
    all_boxes, all_indices = [], []
    for image_index in range(batch):
        first = len(all_boxes)
        for category in np.unique(labels_cpu[image_index]):
            if category == 0:
                continue
            binary = (labels_cpu[image_index] == category).astype(np.uint8)
            count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for component in range(1, count):
                x, y, box_w, box_h, area = stats[component]
                if area < min_area:
                    continue
                dx, dy = float(box_w) * margin, float(box_h) * margin
                all_boxes.append((max(0., x - dx), max(0., y - dy),
                                  min(float(width), x + box_w + dx),
                                  min(float(height), y + box_h + dy)))
                all_indices.append(image_index)
        if len(all_boxes) == first:
            all_boxes.append((0., 0., float(width), float(height)))
            all_indices.append(image_index)
    return (torch.tensor(all_boxes, device=labels.device, dtype=torch.float32),
            torch.tensor(all_indices, device=labels.device, dtype=torch.int64))


def _normalization(channels, device, mean, std):
    if mean is None:
        if channels == 1:
            mean = (.5,)
        elif channels == 3:
            mean = (.485, .456, .406)
        else:
            raise ValueError("Provide mean/std for native images with channel counts other than 1 or 3")
    if std is None:
        std = (.5,) if channels == 1 else ((.229, .224, .225) if channels == 3 else None)
        if std is None:
            raise ValueError("Provide mean/std for this channel count")
    for value, name in ((mean, "mean"), (std, "std")):
        if not isinstance(value, Tensor):
            values = [value] * channels if isinstance(value, (int, float)) else list(value)
            if len(values) != channels or any(not math.isfinite(v) for v in values):
                raise ValueError(f"{name} must have one finite value per channel")
            if name == "std" and any(v <= 0 for v in values):
                raise ValueError("Normalization std must be positive")
    mean_t = torch.as_tensor(mean, device=device, dtype=torch.float32).detach().reshape(-1)
    std_t = torch.as_tensor(std, device=device, dtype=torch.float32).detach().reshape(-1)
    if mean_t.numel() == 1:
        mean_t = mean_t.expand(channels)
    if std_t.numel() == 1:
        std_t = std_t.expand(channels)
    if mean_t.shape != (channels,) or std_t.shape != (channels,):
        raise ValueError("Normalization requires one mean and std per channel")
    # Tensor-valued normalization is device-resident. Its finite/positive values
    # are a caller contract; inspecting their values here would synchronize GPU.
    return mean_t.contiguous(), std_t.contiguous()


def _points(boxes, output_size):
    height, width = output_size
    tx = (torch.arange(width, device=boxes.device, dtype=torch.float32) + .5) / width
    ty = (torch.arange(height, device=boxes.device, dtype=torch.float32) + .5) / height
    x = boxes[:, 0, None, None] + tx[None, None] * (boxes[:, 2] - boxes[:, 0])[:, None, None] - .5
    y = boxes[:, 1, None, None] + ty[None, :, None] * (boxes[:, 3] - boxes[:, 1])[:, None, None] - .5
    return torch.broadcast_tensors(x, y)


def _safe_boxes(boxes):
    boxes = boxes.detach().float()
    ok = torch.isfinite(boxes).all(1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    safe = torch.where(ok[:, None], boxes, boxes.new_tensor((0., 0., 1., 1.)))
    return safe, ok


def _roi_indices(boxes, batch_indices, batch, device):
    _check_boxes(boxes, boxes.shape[0], device)
    if batch_indices is None:
        if boxes.shape[0] != batch:
            raise ValueError("batch_indices are required for multiple crops per image")
        return torch.arange(batch, device=device, dtype=torch.int64)
    if (batch_indices.shape != (boxes.shape[0],) or batch_indices.dtype != torch.int64
            or batch_indices.device != device):
        raise ValueError("batch_indices must be int64 [N] on the image device")
    return batch_indices


def _sample_indexed(source, x, y, indices):
    """Gather only four samples per output pixel, without repeating full images."""
    _, channels, height, width = source.shape
    x, y = x.clamp(0, width - 1), y.clamp(0, height - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
    wx = x - x0 if width > 1 else torch.zeros_like(x)
    wy = y - y0 if height > 1 else torch.zeros_like(y)
    channel = torch.arange(channels, device=source.device)[None, :, None, None]
    offset = (indices[:, None, None, None] * channels + channel) * height * width
    flat = source.reshape(-1)

    def read(xx, yy):
        return flat[offset + (yy * width + xx)[:, None]].float()

    return (read(x0, y0) * ((1 - wy) * (1 - wx))[:, None]
            + read(x1, y0) * ((1 - wy) * wx)[:, None]
            + read(x0, y1) * (wy * (1 - wx))[:, None]
            + read(x1, y1) * (wy * wx)[:, None])


@torch.no_grad()
def paste_probabilities(base_probs: Tensor, crop_probs: Tensor, boxes: Tensor,
                        canvas_size: Tuple[int, int], backend: str = "reference",
                        native_sizes: Optional[Tensor] = None):
    """Map crop probabilities to actual native size; retain round one outside ROI.

    A native pixel is covered if its center falls inside the continuous box.
    Padded canvas pixels are zero and never covered. Probability interpolation is
    FP32 and precedes any threshold/argmax; output dtype matches base_probs.
    Inverse augmentation is only needed for training crop, not inference paste.
    """
    _check_backend(backend)
    canvas_h, canvas_w = _size(canvas_size, "canvas_size")
    for value in (base_probs, crop_probs):
        _check_logits(value)
    if base_probs.shape[:2] != crop_probs.shape[:2] or base_probs.device != crop_probs.device:
        raise ValueError("Base and crop probabilities must share batch/channels/device")
    batch, _, base_h, base_w = base_probs.shape
    _check_boxes(boxes, batch, base_probs.device)
    if native_sizes is None:
        native_sizes = torch.tensor((canvas_h, canvas_w), device=base_probs.device, dtype=torch.int64)[None].expand(batch, 2)
    if native_sizes.shape != (batch, 2) or native_sizes.device != base_probs.device or native_sizes.dtype != torch.int64:
        raise ValueError("native_sizes must be int64 [B,2] on the probability device")
    if backend == "cuda":
        from rescale.ops import cuda_paste_probabilities
        return cuda_paste_probabilities(base_probs, crop_probs, boxes, (canvas_h, canvas_w), native_sizes)
    safe_boxes, boxes_ok = _safe_boxes(boxes)
    sizes_ok = ((native_sizes[:, 0] > 0) & (native_sizes[:, 1] > 0)
                & (native_sizes[:, 0] <= canvas_h) & (native_sizes[:, 1] <= canvas_w))
    sizes = torch.stack((native_sizes[:, 0].clamp(1, canvas_h), native_sizes[:, 1].clamp(1, canvas_w)), 1)
    px = torch.arange(canvas_w, device=base_probs.device, dtype=torch.float32)[None, None, :]
    py = torch.arange(canvas_h, device=base_probs.device, dtype=torch.float32)[None, :, None]
    bx = (px + .5) * (base_w / sizes[:, 1].float())[:, None, None]
    by = (py + .5) * (base_h / sizes[:, 0].float())[:, None, None]
    bx, by = torch.broadcast_tensors(bx, by)
    native_valid = (sizes_ok[:, None, None] & (px < sizes[:, 1, None, None]) & (py < sizes[:, 0, None, None]))
    coverage = (native_valid & boxes_ok[:, None, None]
                & (bx >= safe_boxes[:, 0, None, None]) & (bx < safe_boxes[:, 2, None, None])
                & (by >= safe_boxes[:, 1, None, None]) & (by < safe_boxes[:, 3, None, None]))
    base_grid = torch.stack((2 * bx / base_w - 1, 2 * by / base_h - 1), -1)
    crop_grid = torch.stack((2 * (bx - safe_boxes[:, 0, None, None]) / (safe_boxes[:, 2] - safe_boxes[:, 0])[:, None, None] - 1,
                             2 * (by - safe_boxes[:, 1, None, None]) / (safe_boxes[:, 3] - safe_boxes[:, 1])[:, None, None] - 1), -1)
    base = F.grid_sample(base_probs.float(), base_grid, mode="bilinear", padding_mode="border", align_corners=False)
    crop = F.grid_sample(crop_probs.float(), crop_grid, mode="bilinear", padding_mode="border", align_corners=False)
    output = torch.where(coverage[:, None], crop, base)
    output = torch.where(native_valid[:, None], output, 0.).to(base_probs.dtype)
    return output, coverage[:, None]


@torch.no_grad()
def accumulate_crop_probabilities(crop_probs, boxes, batch_indices, image_size,
                                  native_sizes, sums, counts, backend="reference"):
    """Add cropped probabilities to FP32 canvases; overlaps receive equal votes.

    This operates in place so inference can process crops in bounded batches.
    The reference path samples one ROI at a time. CUDA visits only each ROI's
    native rectangle and atomically adds into the two final canvases.
    """
    _check_backend(backend)
    _check_logits(crop_probs)
    image_h, image_w = _size(image_size, "image_size")
    batch, channels, canvas_h, canvas_w = sums.shape
    if (sums.dtype != torch.float32 or counts.dtype != torch.float32
            or counts.shape != (batch, 1, canvas_h, canvas_w)
            or sums.device != crop_probs.device or counts.device != crop_probs.device
            or crop_probs.shape[1] != channels or crop_probs.shape[0] != boxes.shape[0]):
        raise ValueError("Crop probabilities and FP32 accumulation canvas shapes/devices differ")
    indices = _roi_indices(boxes, batch_indices, batch, sums.device)
    if (native_sizes.shape != (batch, 2) or native_sizes.dtype != torch.int64
            or native_sizes.device != sums.device):
        raise ValueError("native_sizes must be int64 [B,2] on the canvas device")
    if backend == "cuda":
        from rescale.ops import cuda_accumulate_crop_probabilities
        cuda_accumulate_crop_probabilities(crop_probs, boxes, indices, image_size, native_sizes, sums, counts)
        return
    safe_boxes, boxes_ok = _safe_boxes(boxes)
    px = torch.arange(canvas_w, device=sums.device, dtype=torch.float32)[None, :]
    py = torch.arange(canvas_h, device=sums.device, dtype=torch.float32)[:, None]
    for roi in range(boxes.shape[0]):
        image_index = int(indices[roi])
        if image_index < 0 or image_index >= batch:
            continue
        native_h, native_w = native_sizes[image_index]
        size_ok = (native_h > 0) & (native_w > 0) & (native_h <= canvas_h) & (native_w <= canvas_w)
        bx = (px + .5) * (image_w / native_w.clamp_min(1).float())
        by = (py + .5) * (image_h / native_h.clamp_min(1).float())
        bx, by = torch.broadcast_tensors(bx, by)
        box = safe_boxes[roi]
        covered = (boxes_ok[roi] & size_ok & (px < native_w) & (py < native_h)
                   & (bx >= box[0]) & (bx < box[2]) & (by >= box[1]) & (by < box[3]))
        grid = torch.stack((2 * (bx - box[0]) / (box[2] - box[0]) - 1,
                            2 * (by - box[1]) / (box[3] - box[1]) - 1), -1)[None]
        values = F.grid_sample(crop_probs[roi:roi + 1].float(), grid,
                               mode="bilinear", padding_mode="border", align_corners=False)[0]
        sums[image_index].add_(torch.where(covered[None], values, 0.))
        counts[image_index, 0].add_(covered.float())


@torch.no_grad()
def finalize_crop_probabilities(sums, counts, native_sizes, base_probs=None,
                                backend="reference", return_coverage=False):
    """Finalize ROI votes in place, with background outside all expanded boxes.

    sums is an FP32 BCHW canvas; counts is FP32 B1HW and remains unchanged.
    One-channel canvases store foreground probability, so background is zero.
    Multiclass canvases store every class, so background has class 0 equal to 1.
    Optional same-size base_probs retain round one at uncovered native pixels.
    Padding and invalid native sizes always become zero. Nonfinite/nonpositive
    counts are uncovered; finite sums/base values are a caller contract.

    The result aliases sums. CUDA uses one pass and allocates no other full
    probability canvas. Coverage is only returned when requested. In-place sums
    must not overlap counts or base_probs, and CUDA canvases must be contiguous.
    """
    _check_backend(backend)
    _check_logits(sums)
    batch, channels, canvas_h, canvas_w = sums.shape
    if (sums.dtype != torch.float32 or not isinstance(counts, Tensor)
            or counts.shape != (batch, 1, canvas_h, canvas_w)
            or counts.dtype != torch.float32 or counts.device != sums.device):
        raise ValueError("Finalization requires FP32 BCHW sums and B1HW counts on one device")
    if (not isinstance(native_sizes, Tensor) or native_sizes.shape != (batch, 2)
            or native_sizes.dtype != torch.int64 or native_sizes.device != sums.device):
        raise ValueError("native_sizes must be int64 [B,2] on the canvas device")
    if not isinstance(return_coverage, bool):
        raise ValueError("return_coverage must be a boolean")
    if base_probs is not None:
        _check_logits(base_probs)
        if base_probs.shape != sums.shape or base_probs.device != sums.device:
            raise ValueError("Base probabilities must match the sums canvas shape/device")
    for value in (counts, base_probs):
        if value is not None and sums.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
            raise ValueError("In-place sums must not share storage with counts or base probabilities")
    if backend == "cuda":
        from rescale.ops import cuda_finalize_crop_probabilities
        return cuda_finalize_crop_probabilities(sums, counts, native_sizes, base_probs, return_coverage)

    # Small B1HW masks are shared by every class. No clamped count canvas or
    # second BCHW output is needed; uncovered division results are overwritten.
    px = torch.arange(canvas_w, device=sums.device)[None, None, None, :]
    py = torch.arange(canvas_h, device=sums.device)[None, None, :, None]
    native_h = native_sizes[:, 0, None, None, None]
    native_w = native_sizes[:, 1, None, None, None]
    size_ok = (native_h > 0) & (native_w > 0) & (native_h <= canvas_h) & (native_w <= canvas_w)
    native_valid = size_ok & (px < native_w) & (py < native_h)
    coverage = (counts > 0) & torch.isfinite(counts) & native_valid
    missing = native_valid & ~coverage
    sums.div_(counts)
    sums.masked_fill_(~coverage, 0.)
    if base_probs is None:
        if channels > 1:
            sums[:, :1].masked_fill_(missing, 1.)
    else:
        # Copy one class plane at a time to bound reference temporary memory.
        for channel in range(channels):
            plane = sums[:, channel:channel + 1]
            plane.copy_(torch.where(missing, base_probs[:, channel:channel + 1], plane))
    return sums, (coverage if return_coverage else None)


__all__ = ["lesion_boxes_from_logits", "lesion_boxes_from_labels", "paste_probabilities",
           "accumulate_crop_probabilities", "finalize_crop_probabilities"]
