"""Small CPU pools for recursive cache preparation and training."""
def configure_cpu_threads(cfg):
    options = cfg.get("data", {}).get("recursive_cache", {})
    count = options.get("cpu_threads", 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("data.recursive_cache.cpu_threads must be a nonnegative integer")
    if count:
        import cv2
        import torch
        torch.set_num_threads(count)
        cv2.setNumThreads(count)
