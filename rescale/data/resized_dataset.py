"""Bounded resize caches; training batches carry only images, labels and IDs."""
from copy import deepcopy
import time

import numpy as np
import torch

from rescale.data.dataset import SegDataset
from rescale.data.cache import MemoryCache


class RecursiveResizedSegDataset(SegDataset):
    """Reuse ordinary augmentation without retaining decoded original images.

    Only the first deterministic Resize is cached during training. All later
    transforms run on fresh copies on every access, in their original order.
    Validation can cache final tensors because its transforms are deterministic.
    """

    def __init__(self, *args, training_memory_mb=0, validation_memory_mb=0,
                 shared_training_memory_mb=0, training_input_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_memory = MemoryCache(training_memory_mb * 1024 ** 2)
        self.validation_memory = MemoryCache(validation_memory_mb * 1024 ** 2)
        self.shared_training_memory = None
        self._training = False
        self._split_source = self._resize = self._after_resize = None
        self._shared_training_spec = None
        if shared_training_memory_mb > 0:
            from rescale.data.shared_cache import supports_shared_resize_cache
            if supports_shared_resize_cache():
                if training_input_size is None:
                    raise ValueError("Shared resize cache requires training_input_size")
                mask_dtype = np.uint8 if self.mask_mode == "binary" else np.int32
                self._shared_training_spec = (
                    len(self.sample_ids), training_input_size, self.num_channels,
                    mask_dtype, int(shared_training_memory_mb * 1024 ** 2))
                self.training_memory = MemoryCache(0)

    def set_worker_seed(self, seed):
        super().set_worker_seed(seed)
        self._split_source = self._resize = self._after_resize = None

    def _split_transform(self):
        import albumentations as A

        source = self._get_transform()
        if self._split_source is source and self._resize is not None:
            return
        if (not isinstance(source, A.Compose) or not source.transforms
                or not isinstance(source.transforms[0], A.Resize)
                or source.transforms[0].p != 1):
            raise ValueError("Resized recursion requires deterministic Resize first")
        # Preserve the ordinary Compose RNG and the per-transform RNG states.
        # Removing Resize changes no random augmentation or operator ordering.
        after = deepcopy(source)
        self._resize = after.transforms[0]
        after.transforms = after.transforms[1:]
        self._after_resize = after
        self._split_source = source

    def _resized_pair(self, index):
        sample_id = self.sample_ids[index]
        cache = self.shared_training_memory
        resized = cache.get(index) if cache is not None else self.training_memory.get(sample_id)
        if resized is not None:
            return resized
        image, mask = self._load(sample_id)
        result = self._resize(image=image, mask=mask)
        resized = (np.ascontiguousarray(result["image"]),
                   np.ascontiguousarray(result["mask"]), tuple(mask.shape))
        if cache is not None:
            cache.put(index, resized)
        else:
            self.training_memory.put(sample_id, resized)
        return resized

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        if not self._training:
            cached = self.validation_memory.get(sample_id)
            if cached is not None:
                return {"image": cached["image"].clone(), "mask": cached["mask"].clone(), "id": sample_id}
            result = super().__getitem__(index)
            self.validation_memory.put(sample_id, {"image": result["image"], "mask": result["mask"]})
            if sample_id not in self.validation_memory.entries:
                return result
            # DataLoader/caller mutations must not alter cached validation data.
            return {"image": result["image"].clone(), "mask": result["mask"].clone(), "id": sample_id}
        self._split_transform()
        image, mask, _ = self._resized_pair(index)
        result = self._after_resize(image=image.copy(), mask=mask.copy())
        return {"image": result["image"], "mask": torch.as_tensor(result["mask"]).long(), "id": sample_id}

    def enable_training_cache(self):
        """Enable caching of the first resize before workers start."""
        self._training = True
        if self.shared_training_memory is None and self._shared_training_spec is not None:
            from rescale.data.shared_cache import SharedResizeCache
            # Allocate in the parent before workers fork. This is one rank's quota.
            self.shared_training_memory = SharedResizeCache(*self._shared_training_spec)

    def training_cache_stats(self):
        if self.shared_training_memory is not None:
            return self.shared_training_memory.stats()
        return {"cached_samples": len(self.training_memory.entries),
                "capacity_samples": None, "allocated_bytes": self.training_memory.bytes_used}

    def prewarm_training_resizes(self, seed=None):
        """Prepare only entries that fit the manual resized-pixel quota."""
        self.enable_training_cache()
        if self.transform_factory is not None and self._transform_seed is None:
            self.set_worker_seed(0 if seed is None else int(seed))
        self._split_transform()
        cache = self.shared_training_memory
        if cache is not None:
            capacity = cache.capacity
        else:
            h, w = int(self._resize.height), int(self._resize.width)
            mask_bytes = 1 if self.mask_mode == "binary" else 4
            capacity = min(len(self), self.training_memory.limit_bytes // (h * w * (self.num_channels + mask_bytes)))
        started = time.perf_counter()
        for index in range(capacity):
            self._resized_pair(index)
            if index == 0 or (index + 1) % 100 == 0 or index + 1 == capacity:
                print(f"[ResizeRAM] prewarm {index + 1}/{capacity} elapsed={time.perf_counter() - started:.2f}s", flush=True)
        report = self.training_cache_stats()
        report["prepare_seconds"] = time.perf_counter() - started
        return report
