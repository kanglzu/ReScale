"""AMP API compatibility for CUDA training and benchmarks."""
import torch


def make_cuda_grad_scaler(enabled, **options):
    """Use the unified API when available, otherwise use the older CUDA API."""
    amp = getattr(torch, "amp", None)
    scaler_class = getattr(amp, "GradScaler", None)
    if scaler_class is not None:
        return scaler_class("cuda", enabled=enabled, **options)
    return torch.cuda.amp.GradScaler(enabled=enabled, **options)
