"""Crop already normalized model inputs without repeating whole source images."""
import torch

from rescale.recursion.geometry import _check_backend, _normalization, _points, _roi_indices, _safe_boxes, _sample_indexed, _size


@torch.no_grad()
def crop_resized(images, boxes, output_size, masks=None, backend="reference",
                 output_dtype=None, mean=None, std=None, batch_indices=None,
                 return_valid=True, black=None):
    """Bilinear image crops and nearest categorical GT on one resized grid.

    Images are already normalized BCHW floats. Boxes use continuous XYXY edges
    with half-pixel centers. Invalid samples
    contain normalized black, GT 0 and valid=False. No image normalization or
    full-image replication is performed here.
    """
    _check_backend(backend)
    output_size = _size(output_size, "output_size")
    if (not isinstance(images, torch.Tensor) or images.ndim != 4
            or min(images.shape) < 1
            or images.dtype not in (torch.float32, torch.float16, torch.bfloat16)):
        raise ValueError("Resized images must be nonempty BCHW FP32/FP16/BF16")
    if not isinstance(return_valid, bool):
        raise ValueError("return_valid must be a boolean")
    batch, channels, height, width = images.shape
    indices = _roi_indices(boxes, batch_indices, batch, images.device)
    integral_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32,
                       torch.int64, torch.bool)
    if masks is not None and (masks.shape != (batch, height, width)
                              or masks.device != images.device
                              or masks.dtype not in integral_dtypes):
        raise ValueError("Resized masks must be integral [B,H,W] on the image device")
    output_dtype = torch.float32 if output_dtype is None else output_dtype
    if output_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Crop output dtype must be FP32/FP16/BF16")
    if black is None:
        mean_t, std_t = _normalization(channels, images.device, mean, std)
        black = -mean_t / std_t
    elif (black.shape != (channels,) or black.device != images.device
          or black.dtype != torch.float32):
        raise ValueError("Normalized black must be FP32 [C] on the image device")
    if backend == "cuda":
        from rescale.ops import cuda_crop_resized
        return cuda_crop_resized(images, masks, boxes, indices, output_size,
                                 output_dtype, black, return_valid)

    safe_boxes, boxes_ok = _safe_boxes(boxes)
    indices_ok = (indices >= 0) & (indices < batch)
    safe_indices = indices.clamp(0, batch - 1)
    x, y = _points(safe_boxes, output_size)
    valid = ((boxes_ok & indices_ok)[:, None, None]
             & (x >= -.5) & (x < width - .5)
             & (y >= -.5) & (y < height - .5))
    # External boxes may overflow coordinate arithmetic; never gather NaNs.
    x = torch.nan_to_num(x).clamp(0, width - 1)
    y = torch.nan_to_num(y).clamp(0, height - 1)
    values = _sample_indexed(images, x, y, safe_indices)
    values = torch.where(valid[:, None], values, black[None, :, None, None])
    targets = None
    if masks is not None:
        ix = (x + .5).floor().long()
        iy = (y + .5).floor().long()
        positions = safe_indices[:, None, None] * height * width + iy * width + ix
        targets = masks.reshape(-1)[positions].long()
        targets = torch.where(valid, targets, 0)
    return values.to(output_dtype), targets, (valid[:, None] if return_valid else None)
