"""Bounded CPU metric work overlapping GPU validation, with scalar-only results."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from rescale.evaluation.metrics import compute_dice, compute_iou, compute_hd95, compute_recall


def metric_record(prediction, target, base, coverage, num_classes, compute_hd, recursive,
                  coverage_target=None, include_recall=True):
    """Measure overlap, boundaries and proposal coverage on the evaluation grid."""
    record = {"recall": compute_recall(prediction, target, num_classes)} if include_recall else {}
    record["dice"] = compute_dice(prediction, target, num_classes)
    record["iou"] = compute_iou(prediction, target, num_classes)
    if compute_hd:
        record["hd95"] = compute_hd95(prediction, target, num_classes)
    if base is not None:
        same_prediction = base is prediction
        record["base_dice"] = record["dice"] if same_prediction else compute_dice(base, target, num_classes)
        record["base_iou"] = record["iou"] if same_prediction else compute_iou(base, target, num_classes)
        if compute_hd:
            record["base_hd95"] = record["hd95"] if same_prediction else compute_hd95(base, target, num_classes)
        if recursive:
            final_dice, base_dice = record["dice"], record["base_dice"]
            # Diagnostic gains count empty/empty as perfect; primary Dice skips it.
            if not np.isfinite(final_dice) and not np.any(prediction > 0) and not np.any(target > 0):
                final_dice = 1.
            if not np.isfinite(base_dice) and not np.any(base > 0) and not np.any(target > 0):
                base_dice = 1.
            gain = final_dice - base_dice
            if np.isfinite(gain):
                record["gain"] = float(gain)
    if recursive and coverage is not None:
        foreground = (target if coverage_target is None else coverage_target) > 0
        total = int(foreground.sum())
        if total:
            covered = int((coverage.astype(bool, copy=False) & foreground).sum())
            # Match the previous FP32 GPU ratio before converting it to Python.
            record["coverage"] = float(np.float32(covered) / np.float32(total))
            record["complete"] = float(covered == total)
    return record


class MetricQueue:
    """Bound pending samples; worker results never retain segmentation maps."""
    def __init__(self, workers):
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
            raise ValueError("metrics.val_workers must be an integer >= 0")
        self.pool = ThreadPoolExecutor(max_workers=workers) if workers else None
        self.pending = deque()
        self.records = []
        self.limit = max(1, 2 * workers)

    def submit(self, *args):
        if self.pool is None:
            self.records.append(metric_record(*args))
            return
        if len(self.pending) >= self.limit:
            self.records.append(self.pending.popleft().result())
        self.pending.append(self.pool.submit(metric_record, *args))

    def finish(self):
        while self.pending:
            self.records.append(self.pending.popleft().result())
        return self.records

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        if self.pool is not None:
            self.pool.shutdown(wait=True)


def aggregate_records(records, compute_hd, recursive):
    if not records:
        raise ValueError("Expected nonempty validation records")

    def average(key, finite=True):
        values = [record[key] for record in records if key in record]
        if finite:
            values = [value for value in values if np.isfinite(value)]
        else:
            values = [value for value in values if not np.isnan(value)]
        return float(np.mean(values)) if values else (None if finite else float("nan"))

    exact = {key: average(key, finite=False) for key in ("dice", "iou")}
    if compute_hd:
        exact["hd95"] = average("hd95", finite=False)
    extra = {}
    if any("base_dice" in record for record in records):
        if not all("base_dice" in record for record in records):
            raise ValueError("Every validation image must have a first-round prediction")
        extra["base"] = {"dice": average("base_dice", finite=False),
                         "iou": average("base_iou", finite=False)}
        if compute_hd:
            extra["base"]["hd95"] = average("base_hd95", finite=False)
    diagnostics = {"recall": average("recall")}
    if recursive:
        gains = [record["gain"] for record in records if "gain" in record]
        positives = sum("coverage" in record for record in records)
        diagnostics.update({
            "refinement_dice_gain": float(np.mean(gains)) if gains else None,
            "refinement_negative_gain_fraction": float(np.mean(np.asarray(gains) < -1e-7)) if gains else None,
            "refinement_dice_gain_sample_count": len(gains),
            "crop_gt_coverage": average("coverage"),
            "crop_gt_complete_fraction": average("complete"),
            "crop_positive_sample_count": positives,
        })
    return exact, extra, diagnostics
