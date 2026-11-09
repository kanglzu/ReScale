"""Small, explicitly bounded CPU caches owned by a dataset worker."""
from collections import OrderedDict

import numpy as np
import torch


def payload_bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(payload_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(payload_bytes(item) for item in value)
    return 0


class MemoryCache:
    def __init__(self, limit_bytes):
        self.limit_bytes = max(0, int(limit_bytes))
        self.bytes_used = 0
        self.entries = OrderedDict()

    def get(self, key):
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return self.entries[key]

    def put(self, key, value):
        size = payload_bytes(value)
        if size > self.limit_bytes or self.limit_bytes == 0:
            return
        if key in self.entries:
            self.bytes_used -= payload_bytes(self.entries.pop(key))
        while self.entries and self.bytes_used + size > self.limit_bytes:
            _, old = self.entries.popitem(last=False)
            self.bytes_used -= payload_bytes(old)
        self.entries[key] = value
        self.bytes_used += size
