"""Opt-in CUDA components, indexed resized crops and probability accumulation.

Import never compiles an extension. CUDA requests fail explicitly if unavailable;
there is no silent CPU or reference fallback. Build caches use PyTorch's external
extension cache (or the caller's TORCH_EXTENSIONS_DIR).
"""
from __future__ import annotations

import os
import importlib.util
from pathlib import Path
import shlex
import shutil
from time import perf_counter

import torch

_EXTENSION = None
_LOADED_FROM = None


def backend_status():
    cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = shutil.which("nvcc")
    if nvcc is None and cuda_root:
        candidate = Path(cuda_root) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
        if candidate.exists():
            nvcc = str(candidate)
    return {"cuda_runtime": torch.cuda.is_available(), "nvcc": nvcc,
            "loaded": _EXTENSION is not None,
            "requested_library": os.environ.get("RECURSIVE_CUDA_LIBRARY_PATH"),
            "loaded_from": _LOADED_FROM,
            "note": "Use compatible PyTorch/CUDA/C++ compiler and ninja, or an explicitly configured matching prebuilt library."}


def load_cuda_extension(verbose=False):
    global _EXTENSION, _LOADED_FROM
    if _EXTENSION is not None:
        return _EXTENSION
    if not torch.cuda.is_available():
        raise RuntimeError("Recursive CUDA geometry requires a CUDA-enabled PyTorch runtime")
    prebuilt = os.environ.get("RECURSIVE_CUDA_LIBRARY_PATH")
    started = perf_counter()
    rank = os.environ.get("RANK", "0")
    source = "prebuilt" if prebuilt else "jit"
    print(f"[RecursiveCUDA] rank={rank} loading source={source}", flush=True)
    if prebuilt:
        try:
            library = Path(prebuilt).expanduser()
            if not library.is_absolute() or library.suffix.lower() not in {".pyd", ".so"}:
                raise ValueError("Expected an absolute .pyd or .so library path")
            library = library.resolve(strict=True)
            if not library.is_file():
                raise ValueError("The prebuilt library path is not a file")
            specification = importlib.util.spec_from_file_location("recursive_geometry_cuda_v3", library)
            if specification is None or specification.loader is None:
                raise ImportError("Python cannot create an extension loader for this library")
            candidate = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(candidate)
            required = ("crop_resized", "paste_probabilities", "component_stats",
                        "component_boxes", "accumulate_crop_probabilities", "finalize_crop_probabilities")
            if any(not callable(getattr(candidate, name, None)) for name in required):
                raise ImportError("The library does not expose the required recursive geometry API")
            _EXTENSION, _LOADED_FROM = candidate, str(library)
            print(f"[RecursiveCUDA] rank={rank} ready source=prebuilt load_or_build={perf_counter() - started:.3f}s", flush=True)
            return _EXTENSION
        except Exception as exc:
            raise RuntimeError(
                "Cannot load the explicit RECURSIVE_CUDA_LIBRARY_PATH. The library must "
                "be compiled from the matching recursive geometry sources for this Python, "
                "PyTorch, CUDA and platform ABI; no fallback is performed. "
                f"Underlying error: {exc}") from exc
    root = Path(__file__).resolve().parent
    try:
        from torch.utils.cpp_extension import load
        extra_flags = shlex.split(os.environ.get("RECURSIVE_CUDA_EXTRA_FLAGS", ""))
        _EXTENSION = load(
            name="recursive_geometry_cuda_v3",
            sources=[str(root / "binding.cpp"), str(root / "geometry_cuda.cu")],
            extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
            # Nearest-label half-pixel ties require the reference FP32 operation
            # order. Never fuse coordinate multiply/add into a different result.
            extra_cuda_cflags=["-O3", "--fmad=false"] + extra_flags,
            verbose=verbose,
        )
        _LOADED_FROM = "jit"
        print(f"[RecursiveCUDA] rank={rank} ready source=jit load_or_build={perf_counter() - started:.3f}s", flush=True)
    except Exception as exc:
        raise RuntimeError(
            "Cannot load recursive CUDA geometry. Configure a compatible CUDA "
            "toolkit, C++ compiler and ninja, or explicitly select backend='reference'. "
            f"Underlying error: {exc}") from exc
    return _EXTENSION


def _cuda(tensor):
    if not tensor.is_cuda:
        raise ValueError("Explicit CUDA geometry requires CUDA input tensors")


@torch.no_grad()
def cuda_crop_resized(images, masks, boxes, batch_indices, output_size, dtype, black,
                      return_valid=True):
    _cuda(images)
    mask = (torch.empty(0, device=images.device, dtype=torch.int64)
            if masks is None else masks.contiguous())
    output_like = torch.empty(0, device=images.device, dtype=dtype)
    image, target, valid = load_cuda_extension().crop_resized(
        images.contiguous(), mask, boxes.detach().float().contiguous(),
        batch_indices.contiguous(), black.contiguous(), list(output_size),
        output_like, return_valid)
    return image, (None if masks is None else target), (valid if return_valid else None)


@torch.no_grad()
def cuda_paste_probabilities(base, crop, boxes, canvas_size, sizes):
    _cuda(base)
    return tuple(load_cuda_extension().paste_probabilities(
        base.contiguous(), crop.contiguous(), boxes.detach().float().contiguous(),
        sizes.contiguous(), list(canvas_size)))


@torch.no_grad()
def cuda_components(labels, min_area, margin):
    _cuda(labels)
    extension = load_cuda_extension()
    stats = extension.component_stats(labels)
    roots = torch.nonzero(stats[:, 4] >= min_area).flatten()
    return tuple(extension.component_boxes(stats, labels, roots, margin))


@torch.no_grad()
def cuda_accumulate_crop_probabilities(crops, boxes, indices, image_size, sizes, sums, counts):
    _cuda(crops)
    if not sums.is_contiguous() or not counts.is_contiguous():
        raise ValueError("In-place CUDA accumulation canvases must be contiguous")
    load_cuda_extension().accumulate_crop_probabilities(
        crops.contiguous(), boxes.detach().float().contiguous(), indices.contiguous(),
        list(image_size), sizes.contiguous(), sums, counts)


@torch.no_grad()
def cuda_finalize_crop_probabilities(sums, counts, sizes, base_probs, return_coverage):
    _cuda(sums)
    for value in (sums, counts, base_probs):
        if value is not None and not value.is_contiguous():
            raise ValueError("In-place CUDA finalization canvases must be contiguous")
    base = sums.new_empty(0) if base_probs is None else base_probs
    coverage = load_cuda_extension().finalize_crop_probabilities(
        sums, counts, sizes.contiguous(), base, return_coverage)
    return sums, (coverage if return_coverage else None)


__all__ = ["backend_status", "load_cuda_extension", "cuda_crop_resized",
           "cuda_paste_probabilities", "cuda_components", "cuda_accumulate_crop_probabilities",
           "cuda_finalize_crop_probabilities"]
