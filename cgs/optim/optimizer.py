"""
Base Optimizer class.

All optimizers (SGD, Adam, etc.) inherit from this.
"""

from typing import List
from ..tensor.tensor import CGSTensor


class Optimizer:
    """
    Base class for parameter optimizers.

    Args:
        params: Iterable of CGSTensor parameters to optimize.
        lr: Learning rate.
    """

    def __init__(self, params, lr: float = 0.01):
        self.params: List[CGSTensor] = list(params)
        self.lr = lr

    def step(self):
        """Perform a single optimization step. Must be overridden."""
        raise NotImplementedError

    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.params:
            param.zero_grad()

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.lr

    def set_lr(self, lr: float):
        """Set learning rate."""
        self.lr = lr
