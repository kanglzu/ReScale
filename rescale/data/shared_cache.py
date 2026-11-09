"""Bounded, immutable resize slots shared by Linux fork workers in one rank."""
import mmap
import multiprocessing
import sys

import numpy as np


def supports_shared_resize_cache():
    return sys.platform == "linux" and multiprocessing.get_context().get_start_method() == "fork"


class SharedResizeCache:
    def __init__(self, sample_count, image_size, channels, mask_dtype, limit_bytes):
        height, width = map(int, image_size)
        mask_dtype = np.dtype(mask_dtype)
        if min(height, width, channels) < 1 or sample_count < 0 or limit_bytes < 0:
            raise ValueError("Shared resize cache requires positive image dimensions and a nonnegative budget")
        image_bytes = height * width * channels
        mask_bytes = height * width * mask_dtype.itemsize
        # Include native size and publication flags within the manual quota.
        self.slot_bytes = image_bytes + mask_bytes + 2 * 8 + 1
        self.capacity = min(sample_count, int(limit_bytes) // self.slot_bytes)
        self.allocated_bytes = self.capacity * self.slot_bytes
        self.lock = multiprocessing.Lock()
        self.images = self.masks = self.sizes = self.ready = self.buffer = None
        if self.capacity == 0:
            return
        # Anonymous shared mapping: no cache files and no /dev/shm mount quota.
        self.buffer = mmap.mmap(-1, self.allocated_bytes)
        offset = 0
        self.images = np.ndarray((self.capacity, height, width, channels), dtype=np.uint8,
                                 buffer=self.buffer, offset=offset)
        offset += self.capacity * image_bytes
        self.masks = np.ndarray((self.capacity, height, width), dtype=mask_dtype,
                                buffer=self.buffer, offset=offset)
        offset += self.capacity * mask_bytes
        self.sizes = np.ndarray((self.capacity, 2), dtype=np.int64, buffer=self.buffer, offset=offset)
        offset += self.capacity * 2 * 8
        self.ready = np.ndarray(self.capacity, dtype=np.uint8, buffer=self.buffer, offset=offset)
        self.ready.fill(0)

    def get(self, index):
        if index < 0 or index >= self.capacity or not self.ready[index]:
            return None
        with self.lock:
            # Acquire the publication lock before reading across processes.
            # Published slots never change; augmentation receives copies.
            image, mask = self.images[index].view(), self.masks[index].view()
            image.setflags(write=False)
            mask.setflags(write=False)
            return image, mask, tuple(self.sizes[index])

    def put(self, index, value):
        if index < 0 or index >= self.capacity:
            return
        image, mask, native_size = value
        if (image.shape != self.images.shape[1:] or image.dtype != np.uint8
                or mask.shape != self.masks.shape[1:] or mask.dtype != self.masks.dtype
                or len(native_size) != 2 or min(native_size) < 1):
            raise ValueError("Resized pixels, labels or native size do not match the shared cache")
        with self.lock:
            if self.ready[index]:
                return
            self.images[index] = image
            self.masks[index] = mask
            self.sizes[index] = native_size
            self.ready[index] = 1

    def stats(self):
        return {"cached_samples": int(np.count_nonzero(self.ready)) if self.capacity else 0,
                "capacity_samples": self.capacity, "allocated_bytes": self.allocated_bytes}
