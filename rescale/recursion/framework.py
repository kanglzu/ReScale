"""Round orchestration; the shared network lives in models/predictor.py.

forward() is one differentiable pass for sequential training. predict() runs the
complete image -> lesion boxes -> indexed crops -> shared predictor.
"""
import math
import torch
from torch import nn
from rescale.recursion.resized import crop_resized
from rescale.recursion.geometry import lesion_boxes_from_logits, paste_probabilities, accumulate_crop_probabilities, finalize_crop_probabilities, _normalization

def main_logits(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs

def probabilities(outputs):
    logits = main_logits(outputs)
    logits = logits.float()
    if logits.shape[1] < 2:
        raise ValueError('Recursive segmentation uses background + foreground logits')
    return logits.softmax(1)

class RecursiveSegFramework(nn.Module):

    def __init__(self, backbone, input_size, innovations=None, mean=None, std=None):
        super().__init__()
        if not innovations:
            innovations = {}
        options = dict(innovations)
        recursion = dict(options.get('recursion', {}))
        execution = dict(options.get('cuda', {}))
        self.backbone = backbone
        input_dimensions = []
        for dimension in input_size:
            input_dimensions.append(int(dimension))
        self.input_size = tuple(input_dimensions)
        self.recursion_enabled = bool(recursion.get('enabled', True))
        requested_rounds = int(recursion.get('rounds', 2))
        if requested_rounds not in (1, 2):
            raise ValueError('The first implementation supports rounds=1 or 2')
        if self.recursion_enabled:
            self.rounds = requested_rounds
        else:
            self.rounds = 1
        self.margin = float(recursion.get('margin', 0.1))
        self.threshold = float(recursion.get('threshold', 0.5))
        self.crop_source = recursion.get('crop_source', 'resized')
        if self.crop_source != 'resized':
            raise ValueError("ReScale crops from the resized input grid")
        self.loss_weight = float(recursion.get('loss_weight', 1.0))
        self.min_area = recursion.get('min_area', 16)
        self.crop_batch_size = recursion.get('crop_batch_size', 8)
        self.outside_roi_background = recursion.get('outside_roi_background', True)
        if not isinstance(self.outside_roi_background, bool):
            raise ValueError('outside_roi_background must be a boolean')
        if len(self.input_size) != 2 or min(self.input_size) < 1:
            raise ValueError('input_size must be positive [H,W]')
        if not math.isfinite(self.margin) or self.margin < 0 or (not 0 < self.threshold < 1) or (not math.isfinite(self.loss_weight)) or (self.loss_weight < 0):
            raise ValueError('Invalid margin, threshold, or second-round loss weight')
        for value in (self.min_area, self.crop_batch_size):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('min_area and crop_batch_size must be positive integers')
        if execution.get('enabled', False):
            self.backend = 'cuda'
        else:
            self.backend = 'reference'
        self.mean = mean
        self.std = std
        self._resized_black = None
        self._aux = {}

    def forward(self, images, scale=None, stage=1):
        if tuple(images.shape[-2:]) != self.input_size:
            raise ValueError('Input spatial size differs from model.input_size')
        if isinstance(stage, bool) or stage not in (1, 2):
            raise ValueError('Segmentation stage must be 1 or 2')
        return self.backbone(images, scale=scale)

    @torch.no_grad()
    def make_rois(self, outputs):
        logits = main_logits(outputs)
        logits = logits.detach()
        return lesion_boxes_from_logits(logits, min_area=self.min_area, margin=self.margin, threshold=self.threshold, backend=self.backend)

    @torch.no_grad()
    def crop_scale(self, boxes):
        (height, width) = self.input_size
        extent = boxes[:, 2:] - boxes[:, :2]
        image_extent = boxes.new_tensor((width, height))
        relative_extent = extent / image_extent
        relative_extent = relative_extent.clamp_min(1e-08)
        return relative_extent.log()

    @torch.no_grad()
    def crop_resized(self, images, boxes, masks=None, output_dtype=None, batch_indices=None, return_valid=True):
        channels = images.shape[1]
        if self._resized_black is None or self._resized_black.device != images.device or self._resized_black.numel() != channels or (self._resized_black.dtype != torch.float32):
            (mean, std) = _normalization(channels, images.device, self.mean, self.std)
            self._resized_black = -mean / std
        return crop_resized(images, boxes, self.input_size, masks=masks, backend=self.backend, output_dtype=output_dtype, batch_indices=batch_indices, return_valid=return_valid, black=self._resized_black)

    @torch.inference_mode()
    def predict(self, images, native=None, canvas_size=None, diagnostics=False, return_base_canvas=True):
        if self.training:
            raise RuntimeError('Call eval() before predict()')
        if native is not None and 'native_mask' in native:
            raise ValueError('Inference native inputs must not contain ground truth')
        self.clear_aux_outputs()
        base_outputs = self(images)
        base_logits = main_logits(base_outputs)
        base_input_logits = base_logits if diagnostics else None
        base_input_probabilities = None
        crop_dtype = base_logits.dtype
        canvas_size = tuple(canvas_size or self.input_size)
        if len(canvas_size) != 2 or any((not isinstance(value, int) or value < 1 for value in canvas_size)):
            raise ValueError('canvas_size must contain two positive integer dimensions')
        (batch_size, channels, height, width) = base_logits.shape
        device = base_logits.device
        sizes = torch.tensor(canvas_size, device=device, dtype=torch.int64)[None].expand(batch_size, 2).contiguous()
        base_canvas = None
        coverage = None
        needs_base_canvas = self.rounds == 1 or not self.outside_roi_background or (diagnostics and return_base_canvas)
        if needs_base_canvas or diagnostics:
            base = probabilities(base_logits)
            if diagnostics:
                base_input_probabilities = base
        if needs_base_canvas:
            full_box = (0.0, 0.0, float(width), float(height))
            full_boxes = base.new_tensor(full_box).expand(batch_size, 4)
            if canvas_size == self.input_size:
                base_canvas = base
                if self.rounds == 1 and diagnostics:
                    coverage = torch.ones((batch_size, 1, height, width), device=device, dtype=torch.bool)
            else:
                (base_canvas, coverage) = paste_probabilities(base, base, full_boxes, canvas_size, backend=self.backend, native_sizes=sizes)
            if self.rounds == 2 or not diagnostics:
                coverage = None
        if needs_base_canvas or diagnostics:
            del base
        del base_outputs
        self.clear_aux_outputs()
        roi_counts = None
        if diagnostics:
            roi_counts = torch.zeros(batch_size, device=device, dtype=torch.long)
        second_round_batches = 0
        if self.rounds == 2:
            (boxes, batch_indices) = self.make_rois(base_logits)
            del base_logits
            if diagnostics:
                roi_counts.scatter_add_(0, batch_indices, torch.ones_like(batch_indices))
            sums = torch.zeros((batch_size, channels, *canvas_size), device=device, dtype=torch.float32)
            counts = torch.zeros((batch_size, 1, *canvas_size), device=device, dtype=torch.float32)
            coverage = None
            for start in range(0, boxes.shape[0], self.crop_batch_size):
                stop = start + self.crop_batch_size
                crop_boxes = boxes[start:stop]
                crop_indices = batch_indices[start:stop]
                crops = self.crop_resized(images, crop_boxes, output_dtype=crop_dtype, batch_indices=crop_indices, return_valid=False)[0]
                scale = self.crop_scale(crop_boxes)
                refined_outputs = self(crops, scale=scale)
                refined = probabilities(refined_outputs)
                accumulate_crop_probabilities(refined, crop_boxes, crop_indices, self.input_size, sizes, sums, counts, backend=self.backend)
                second_round_batches += 1
                del crops, refined_outputs, refined, scale
                self.clear_aux_outputs()
            fallback = None if self.outside_roi_background else base_canvas
            (result, coverage) = finalize_crop_probabilities(sums, counts, sizes, base_probs=fallback, backend=self.backend, return_coverage=diagnostics)
        else:
            del base_logits
            boxes = full_boxes
            batch_indices = torch.arange(batch_size, device=device)
            result = base_canvas
        if diagnostics:
            self._aux = {'base_input_logits': base_input_logits, 'base_input_probabilities': base_input_probabilities, 'base_probabilities': base_canvas, 'refined_probabilities': result, 'boxes': boxes, 'batch_indices': batch_indices, 'roi_counts': roi_counts, 'coverage': coverage, 'second_round_batches': second_round_batches}
        return result

    def get_aux_outputs(self):
        return self._aux

    def clear_aux_outputs(self):
        self._aux = {}
        clear_backbone = getattr(self.backbone, 'clear_aux_outputs', None)
        if clear_backbone is not None:
            clear_backbone()
