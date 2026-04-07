"""
DataLoader — Mini-batch iteration with shuffling.
"""

import numpy as np
from typing import Optional


class DataLoader:
    """
    Iterates over a dataset in mini-batches.

    Args:
        dataset: A Dataset instance.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data at each epoch.
        drop_last: Whether to drop the last incomplete batch.
    """

    def __init__(self, dataset, batch_size: int = 32, shuffle: bool = True,
                 drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        n = len(self.dataset)
        indices = np.arange(n)

        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            if self.drop_last and (end - start) < self.batch_size:
                break

            batch_indices = indices[start:end]
            batch_x = []
            batch_y = []

            for idx in batch_indices:
                x, y = self.dataset[idx]
                batch_x.append(x)
                batch_y.append(y)

            yield np.array(batch_x, dtype=np.float64), np.array(batch_y)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
