"""Reproducibility — Set random seeds."""

import numpy as np


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    print(f"  🎲 Random seed set to {seed}")
