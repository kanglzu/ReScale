"""
PyTorch Dataset for 2D image segmentation.

Reads image/label pairs listed in manifest.json for a given set of sample ids.
Works with any file format readable by OpenCV (PNG, JPG, BMP, TIFF, …).
"""
from typing import Callable, Dict, List, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SegDataset(Dataset):
    """2D segmentation dataset backed by a manifest.json sample list.

    Args:
        sample_ids: List of sample ids to include.
        sample_dict: Mapping from sample id to manifest entry
            (output of data.split.get_sample_dict).
        transform: albumentations Compose transform applied to (image, mask).
        num_channels: 1 for grayscale, 3 for RGB.
    """

    def __init__(self, sample_ids: List[str], sample_dict: Dict[str, dict], transform: Optional[Callable]=None, transform_factory: Optional[Callable[[int], Callable]]=None, num_channels: int=3, transform_seed: Optional[int]=None, mask_mode: str='binary', num_classes: Optional[int]=None):
        if transform is not None and transform_factory is not None:
            raise ValueError('Specify either transform or transform_factory, not both.')
        self.sample_ids = sample_ids
        self.sample_dict = sample_dict
        self.transform = transform
        self.transform_factory = transform_factory
        self.num_channels = num_channels
        if mask_mode not in {'binary', 'multiclass'}:
            raise ValueError('mask_mode must be binary or multiclass')
        self.mask_mode = mask_mode
        self.num_classes = num_classes
        self._transform_seed = transform_seed
        self._transform_instance = transform

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        image, mask = self._load(sid)
        transform = self._get_transform()
        if transform is not None:
            result = transform(image=image, mask=mask)
            (image, mask) = (result['image'], result['mask'])
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        return {'image': image, 'mask': mask.long(), 'id': sid}

    def _load(self, sid: str):
        """Load a single (image, mask) pair from disk."""
        entry = self.sample_dict[sid]
        image = _read_image(entry['image'], self.num_channels)
        mask = _read_mask(entry['label'], mask_mode=self.mask_mode, num_classes=self.num_classes)
        return (image, mask)

    def set_worker_seed(self, seed: int) -> None:
        """Bind a deterministic transform seed to this dataset copy."""
        self._transform_seed = int(seed)
        if self.transform_factory is not None:
            self._transform_instance = None

    def _get_transform(self) -> Optional[Callable]:
        """Resolve the transform for the current dataset copy."""
        if self.transform_factory is None:
            return self.transform
        if self._transform_instance is None:
            if self._transform_seed is None:
                raise RuntimeError('transform_factory requires a worker seed before first use.')
            self._transform_instance = self.transform_factory(int(self._transform_seed))
        return self._transform_instance

def _read_image(path: str, num_channels: int) -> np.ndarray:
    """Read image as (H, W, C) uint8 numpy array.

    Args:
        path: File path.
        num_channels: 1 → grayscale output, 3 → RGB output.

    Returns:
        uint8 numpy array of shape (H, W, C).
    """
    flag = cv2.IMREAD_GRAYSCALE if num_channels == 1 else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    if num_channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif num_channels == 1:
        img = img[:, :, np.newaxis]
    return img

def _read_mask(path: str, *, mask_mode: str='binary', num_classes: Optional[int]=None) -> np.ndarray:
    """Read a label plane, keeping legacy binary mapping unless explicitly changed.

    In binary mode any non-zero pixel is mapped to foreground label 1. This supports
    datasets stored as 0/255 masks and instance-id masks such as 0/1/2/3/4.

    Args:
        path: File path.

    Returns:
        Binary uint8 {0,1}, or multiclass int32 with preserved integer class IDs.
    """
    if mask_mode == 'multiclass':
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f'Cannot read mask: {path}')
        if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f'Multiclass masks need a single integer label plane: {path}')
        if num_classes is None or num_classes < 2:
            raise ValueError('Multiclass mask loading requires num_classes >= 2')
        if mask.min() < 0 or mask.max() >= num_classes:
            raise ValueError(f'Mask labels outside [0, {num_classes - 1}]: {path}')
        return mask.astype(np.int32)
    if mask_mode != 'binary':
        raise ValueError('mask_mode must be binary or multiclass')
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f'Cannot read mask: {path}')
    return (mask > 0).astype(np.uint8)
