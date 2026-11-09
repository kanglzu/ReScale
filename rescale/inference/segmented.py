"""Opt-in batch-one graph segments for the existing resized CUDA predictor.

All eligible components remain present and in their original order. Dynamic
nonzero remains outside capture; no GT, ROI limit, or new resampling is used.
This session changes no parameter/checkpoint and is never enabled by training.
"""
from contextlib import AbstractContextManager, ExitStack
import math
import threading

import torch

from rescale.recursion.framework import main_logits, probabilities
from rescale.recursion.geometry import accumulate_crop_probabilities, finalize_crop_probabilities
from rescale.inference.graph import InferenceGraphSession
from rescale.ops import load_cuda_extension


class _Coarse(torch.nn.Module):
    def __init__(self, forward, extension, threshold, min_area):
        super().__init__()
        self.forward_network = forward
        self.extension = extension
        self.cutoff = math.log(threshold / (1 - threshold))
        self.min_area = min_area

    def forward(self, images):
        logits = main_logits(self.forward_network(images))
        values = logits.detach().float()
        if values.shape[1] != 2:
            raise ValueError("Segmented capture currently supports two-class binary models")
        labels = ((values[:, 1] - values[:, 0]) >= self.cutoff).int()
        stats = self.extension.component_stats(labels.contiguous())
        return logits, probabilities(logits), labels, stats, stats[:, 4] >= self.min_area


class _Refine(torch.nn.Module):
    def __init__(self, framework, forward, diagnostics):
        super().__init__()
        object.__setattr__(self, 'framework', framework)
        self.forward_network = forward
        self.diagnostics = diagnostics
        self.sizes = torch.tensor([framework.input_size], device=next(framework.parameters()).device, dtype=torch.int64)
        height, width = framework.input_size
        self.image_extent = torch.tensor((width, height), device=self.sizes.device, dtype=torch.float32)

    def forward(self, images, scale=None):
        boxes = scale
        framework = self.framework
        indices = torch.zeros(len(boxes), device=boxes.device, dtype=torch.int64)
        height, width = framework.input_size
        sums = torch.zeros((1, 2, height, width), device=images.device, dtype=torch.float32)
        counts = torch.zeros((1, 1, height, width), device=images.device, dtype=torch.float32)
        logits_list = []
        for start in range(0, len(boxes), framework.crop_batch_size):
            crop_boxes = boxes[start:start + framework.crop_batch_size]
            crop_indices = indices[start:start + framework.crop_batch_size]
            crops = framework.crop_resized(images, crop_boxes, output_dtype=torch.bfloat16,
                batch_indices=crop_indices, return_valid=False)[0]
            # Keep the image extent on device during capture.
            extent = crop_boxes[:, 2:] - crop_boxes[:, :2]
            crop_scale = (extent / self.image_extent).clamp_min(1e-8).log()
            logits = main_logits(self.forward_network(crops, scale=crop_scale))
            crop_probs = probabilities(logits)
            accumulate_crop_probabilities(crop_probs, crop_boxes, crop_indices,
                framework.input_size, self.sizes, sums, counts, backend='cuda')
            logits_list.append(logits)
        result, coverage = finalize_crop_probabilities(sums, counts, self.sizes,
            base_probs=None, backend='cuda', return_coverage=self.diagnostics)
        return result, coverage, logits_list


class SegmentedRecursiveSession(AbstractContextManager):
    """Graph replay, B=1 resized/CUDA/shared predictor; other cases use original.

    The graph cache has at most max_graphs shape/policy entries per segment.
    Input and output copies are included. Calls must remain on one thread and
    stream with gradients disabled. Load checkpoints before entering, then keep
    weights, backend flags and precision fixed until exit. A session is not a
    replacement for equivalence checks in each intended deployment environment.
    """
    def __init__(self, framework, *, prepare_weights=False, cache_amp_weights=False, max_graphs=8, max_graph_rois=24):
        self.framework = framework
        if prepare_weights or cache_amp_weights:
            raise ValueError("Weight preparation candidates are not approved; retain original parameters")
        self.max_graphs = max_graphs
        if max_graph_rois < 1:
            raise ValueError("max_graph_rois must be positive")
        self.max_graph_rois = max_graph_rois
        self.last_refinement_logits = []

    def __enter__(self):
        m = self.framework
        if (m.training or torch.is_grad_enabled() or m.crop_source != 'resized'
                or m.backend != 'cuda' or m.rounds != 2
                or not m.outside_roi_background
                or getattr(m, 'sampling_strategy', 'crop') != 'crop'
                or getattr(m.backbone, 'num_classes', None) != 2):
            raise ValueError("Segmented inference requires eval/no_grad, resized CUDA, shared binary two-round model")
        if getattr(m, '_segmented_inference_active', False):
            raise RuntimeError("A segmented inference session is already active")
        self.stack = ExitStack()
        try:
            forward = m.backbone.forward
            refinement_forward = forward
            extension = load_cuda_extension()
            self.coarse = _Coarse(forward, extension, m.threshold, m.min_area).eval()
            self.coarse_session = self.stack.enter_context(InferenceGraphSession(
                self.coarse, uses_scale=False, max_graphs=self.max_graphs))
            self.tails = {}
            self._tail_forwards = {}
            self._original = m.predict
            self._had_predict = 'predict' in m.__dict__
            self._previous_predict = m.__dict__.get('predict')
            m.predict = self.predict
            m._segmented_inference_active = True
            self._extension = extension
            self._forward_network = refinement_forward
            self._owner = threading.get_ident()
            return self
        except BaseException:
            self.stack.close()
            raise

    @torch.inference_mode()
    def predict(self, images, native=None, canvas_size=None, diagnostics=False, return_base_canvas=True):
        m = self.framework
        if threading.get_ident() != self._owner:
            raise RuntimeError("Use a separate segmented session per thread")
        if (len(images) != 1 or not images.is_cuda or tuple(images.shape[-2:]) != m.input_size
                or native is not None or (canvas_size and tuple(canvas_size) != m.input_size)
                or not torch.is_autocast_enabled() or torch.get_autocast_gpu_dtype() != torch.bfloat16):
            return self._original(images, native=native, canvas_size=canvas_size,
                                  diagnostics=diagnostics, return_base_canvas=return_base_canvas)
        if m.training or torch.is_grad_enabled():
            raise RuntimeError("Segmented inference supports frozen evaluation only")
        m.clear_aux_outputs()
        logits, base, labels, stats, keep = self.coarse(images)
        roots = torch.nonzero(keep).flatten()
        if roots.numel():
            # All source indices equal zero at B=1, so the original stable sort
            # and missing-image append leave exactly this order unchanged.
            boxes, indices = self._extension.component_boxes(stats, labels, roots, m.margin)
        else:
            height, width = m.input_size
            boxes = logits.new_tensor([[0., 0., float(width), float(height)]], dtype=torch.float32)
            indices = torch.zeros(1, device=images.device, dtype=torch.int64)
        if diagnostics not in self.tails:
            tail = _Refine(m, self._forward_network, diagnostics).eval()
            self._tail_forwards[diagnostics] = tail.forward
            self.stack.enter_context(InferenceGraphSession(tail, max_graphs=self.max_graphs))
            self.tails[diagnostics] = tail
        if len(boxes) > self.max_graph_rois:
            # This limits cached workspace, never the number of refined ROIs.
            result, coverage, self.last_refinement_logits = self._tail_forwards[diagnostics](images, scale=boxes)
        else:
            result, coverage, self.last_refinement_logits = self.tails[diagnostics](images, scale=boxes)
        if diagnostics:
            m._aux = dict(base_input_logits=logits, base_input_probabilities=base,
                base_probabilities=base if return_base_canvas else None,
                refined_probabilities=result, boxes=boxes, batch_indices=indices,
                roi_counts=torch.full((1,), len(boxes), device=images.device, dtype=torch.int64),
                coverage=coverage, second_round_batches=math.ceil(len(boxes) / m.crop_batch_size))
        return result

    def __exit__(self, exc_type, exc_value, traceback):
        m = self.framework
        if self._had_predict:
            m.predict = self._previous_predict
        else:
            del m.predict
        del m._segmented_inference_active
        try:
            return self.stack.__exit__(exc_type, exc_value, traceback)
        finally:
            self.last_refinement_logits = []
