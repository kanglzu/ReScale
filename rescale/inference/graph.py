"""Opt-in CUDA Graph replay for a frozen, single-stream evaluation session.

Use after loading weights and calling eval(). Do not load weights, change device,
precision/backend flags, or mutate parameters while the session is active.
This helper is not installed by the trainer and changes no checkpoint keys.
"""

from contextlib import AbstractContextManager
import threading

import torch




class InferenceGraphSession(AbstractContextManager):
    """Temporarily replay a tensor-returning backbone, with bounded shape cache.

    The module must accept ``forward(images, scale=None)`` when uses_scale=True.
    Output tensors are cloned, so future replay cannot overwrite saved logits.
    Uncached shapes beyond max_graphs/max_batch use the original eager forward.
    Capture/warmup is lazy and must be excluded from steady-state measurements.
    A session belongs to its entering thread and one CUDA stream. Exit before
    training, loading a checkpoint, modifying weights or changing runtime flags.
    """

    def __init__(self, module, *, uses_scale=True, max_graphs=8, max_batch=32):
        if max_graphs < 1 or max_batch < 1:
            raise ValueError("max_graphs and max_batch must be positive")
        self.module = module
        self.uses_scale = uses_scale
        self.max_graphs = int(max_graphs)
        self.max_batch = int(max_batch)
        self.cache = {}
        self.hits = self.fallbacks = 0
        self._active = False

    def __enter__(self):
        if self._active or getattr(self.module, "_inference_graph_active", False):
            raise RuntimeError("A graph session is already active")
        if any(m.training for m in self.module.modules()):
            raise RuntimeError("Call eval() before entering a graph session")
        if torch.is_grad_enabled():
            raise RuntimeError("Use torch.no_grad() or torch.inference_mode()")
        self._owner = threading.get_ident()
        self._original = self.module.forward
        self._had_forward = "forward" in self.module.__dict__
        self._previous_forward = self.module.__dict__.get("forward")
        self._stream = None
        self.module._inference_graph_active = True
        self.module.forward = self._forward
        self._active = True
        return self

    def _call(self, x, scale):
        if self.uses_scale:
            return self._original(x, scale=scale)
        return self._original(x)

    @staticmethod
    def _signature(x):
        return (tuple(x.shape), tuple(x.stride()), x.dtype, x.device)

    @staticmethod
    def _clone_output(value):
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, tuple):
            return tuple(InferenceGraphSession._clone_output(v) for v in value)
        if isinstance(value, list):
            return [InferenceGraphSession._clone_output(v) for v in value]
        if value is None:
            return None
        raise TypeError("Graph outputs must be tensors, tuples/lists of tensors, or None")

    def _forward(self, x, scale=None):
        if threading.get_ident() != self._owner:
            raise RuntimeError("Graph sessions cannot be shared across threads")
        if self.module.training or torch.is_grad_enabled():
            raise RuntimeError("Graph replay supports frozen evaluation only")
        if not x.is_cuda or len(x) > self.max_batch:
            self.fallbacks += 1
            return self._call(x, scale)
        current = torch.cuda.current_stream(x.device)
        stream_key = (x.device, current.cuda_stream)
        if self._stream is None:
            self._stream = stream_key
        if self._stream != stream_key:
            raise RuntimeError("Use a separate graph session per CUDA stream")
        if scale is not None and (not scale.is_cuda or scale.device != x.device):
            self.fallbacks += 1
            return self._call(x, scale)
        amp = torch.is_autocast_enabled()
        amp_dtype = torch.get_autocast_gpu_dtype()
        key = (self._signature(x), None if scale is None else self._signature(scale),
               amp, amp_dtype, torch.is_inference_mode_enabled())
        if key not in self.cache:
            if len(self.cache) >= self.max_graphs:
                self.fallbacks += 1
                return self._call(x, scale)
            sx = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)
            sx.copy_(x)
            ss = None if scale is None else torch.empty_strided(
                scale.shape, scale.stride(), device=scale.device, dtype=scale.dtype)
            if ss is not None:
                ss.copy_(scale)
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream), torch.autocast(
                    "cuda", enabled=amp, dtype=amp_dtype, cache_enabled=False):
                for _ in range(3):
                    output = self._call(sx, ss)
            current.wait_stream(stream)
            torch.cuda.synchronize(x.device)
            del output
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.autocast(
                    "cuda", enabled=amp, dtype=amp_dtype, cache_enabled=False):
                output = self._call(sx, ss)
            # Validate the return structure before caching persistent buffers.
            self._clone_output(output)
            self.cache[key] = (graph, sx, ss, output)
        graph, sx, ss, output = self.cache[key]
        sx.copy_(x)
        if ss is not None:
            ss.copy_(scale)
        graph.replay()
        self.hits += 1
        return self._clone_output(output)

    def stats(self):
        return {"graphs": len(self.cache), "hits": self.hits,
                "eager_fallbacks": self.fallbacks, "max_graphs": self.max_graphs,
                "max_batch": self.max_batch}

    def __exit__(self, exc_type, exc_value, traceback):
        if self._had_forward:
            self.module.forward = self._previous_forward
        else:
            del self.module.forward
        del self.module._inference_graph_active
        self.cache.clear()
        self._active = False
        return False
