"""
Dataset classes for loading and managing data.

Provides a base Dataset class and concrete implementations for common
benchmarks (MNIST, CIFAR-10) with automatic download.
"""

import numpy as np
import os
import gzip
import struct
import ssl
import urllib.request
from typing import Tuple, Optional

# Fix SSL certificate issues on some systems
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass


class Dataset:
    """
    Base dataset class.

    Subclasses must implement __getitem__ and __len__.
    """

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class MNISTDataset(Dataset):
    """
    MNIST handwritten digits dataset.

    Automatically downloads and parses the IDX binary files.

    Args:
        root: Directory to store downloaded data.
        train: If True, load training data; else load test data.
        normalize: If True, normalize pixel values to [0, 1].
        flatten: If True, flatten 28x28 images to 784-dim vectors.
        subset_fraction: Fraction of data to use (for data-efficiency experiments).
    """

    # File names for MNIST IDX files
    FILES = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz',
    }

    # Mirror URLs to try in order (primary → fallbacks)
    MIRRORS = [
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
        'https://storage.googleapis.com/cvdf-datasets/mnist/',
        'http://yann.lecun.com/exdb/mnist/',
    ]

    def __init__(self, root: str = './data/mnist', train: bool = True,
                 normalize: bool = True, flatten: bool = True,
                 subset_fraction: float = 1.0):
        self.root = root
        self.train = train
        self.normalize = normalize
        self.flatten = flatten

        os.makedirs(root, exist_ok=True)

        if train:
            self.images = self._load_images('train_images')
            self.labels = self._load_labels('train_labels')
        else:
            self.images = self._load_images('test_images')
            self.labels = self._load_labels('test_labels')

        if normalize:
            self.images = self.images.astype(np.float64) / 255.0

        if flatten:
            self.images = self.images.reshape(self.images.shape[0], -1)

        # Subset for data-efficiency experiments
        if subset_fraction < 1.0:
            n = int(len(self.images) * subset_fraction)
            indices = np.random.choice(len(self.images), n, replace=False)
            self.images = self.images[indices]
            self.labels = self.labels[indices]

    def _download(self, key: str) -> str:
        """Download a file if not already cached, trying multiple mirrors."""
        fname = self.FILES[key]
        filepath = os.path.join(self.root, fname)
        if os.path.exists(filepath):
            return filepath

        for mirror in self.MIRRORS:
            url = mirror + fname
            try:
                print(f"Downloading {url}...")
                urllib.request.urlretrieve(url, filepath)
                return filepath
            except Exception as e:
                print(f"  Mirror failed ({e}), trying next...")
                if os.path.exists(filepath):
                    os.remove(filepath)

        raise RuntimeError(
            f"Failed to download {fname} from all mirrors. "
            f"You can manually download MNIST files and place them in {self.root}"
        )

    def _load_images(self, key: str) -> np.ndarray:
        """Load IDX image file."""
        path = self._download(key)
        with gzip.open(path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>4I', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num, rows, cols)
        return images

    def _load_labels(self, key: str) -> np.ndarray:
        """Load IDX label file."""
        path = self._download(key)
        with gzip.open(path, 'rb') as f:
            magic, num = struct.unpack('>2I', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self.images[index], int(self.labels[index])

    def __len__(self) -> int:
        return len(self.images)

    def __repr__(self):
        return (f"MNISTDataset(split={'train' if self.train else 'test'}, "
                f"size={len(self)}, flatten={self.flatten})")


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing and debugging.

    Generates random data with linear or nonlinear decision boundaries.

    Args:
        num_samples: Number of samples.
        input_dim: Input feature dimension.
        num_classes: Number of classes.
        noise: Noise level.
    """

    def __init__(self, num_samples: int = 1000, input_dim: int = 10,
                 num_classes: int = 2, noise: float = 0.1):
        np.random.seed(42)
        self.images = np.random.randn(num_samples, input_dim).astype(np.float64)

        # Create labels based on a random linear boundary
        W = np.random.randn(input_dim, num_classes)
        logits = self.images @ W + noise * np.random.randn(num_samples, num_classes)
        self.labels = np.argmax(logits, axis=1)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self.images[index], int(self.labels[index])

    def __len__(self) -> int:
        return len(self.images)
