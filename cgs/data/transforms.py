"""
Data Transforms for augmentation and preprocessing.

Used by the Multi-View Encoder to create diverse views of input data.
"""

import numpy as np
from typing import List


class Compose:
    """Chain multiple transforms together."""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        return f"Compose({[repr(t) for t in self.transforms]})"


class Normalize:
    """Normalize input by mean and std."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-8)

    def __repr__(self):
        return f"Normalize(mean={self.mean}, std={self.std})"


class AddGaussianNoise:
    """Add Gaussian noise to input."""

    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x + np.random.randn(*x.shape) * self.std

    def __repr__(self):
        return f"AddGaussianNoise(std={self.std})"


class RandomMask:
    """Randomly zero out a fraction of input elements."""

    def __init__(self, ratio: float = 0.15):
        self.ratio = ratio

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mask = np.random.rand(*x.shape) > self.ratio
        return x * mask.astype(x.dtype)

    def __repr__(self):
        return f"RandomMask(ratio={self.ratio})"


class RandomScale:
    """Randomly scale input by a factor."""

    def __init__(self, low: float = 0.8, high: float = 1.2):
        self.low = low
        self.high = high

    def __call__(self, x: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.low, self.high)
        return x * scale

    def __repr__(self):
        return f"RandomScale(low={self.low}, high={self.high})"


class RandomShift:
    """Randomly shift input values."""

    def __init__(self, max_shift: float = 0.1):
        self.max_shift = max_shift

    def __call__(self, x: np.ndarray) -> np.ndarray:
        shift = np.random.uniform(-self.max_shift, self.max_shift)
        return x + shift

    def __repr__(self):
        return f"RandomShift(max_shift={self.max_shift})"
