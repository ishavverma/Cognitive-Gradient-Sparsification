"""
Stochastic Gradient Descent with optional Momentum and Weight Decay.
"""

import numpy as np
from .optimizer import Optimizer


class SGD(Optimizer):
    """
    SGD optimizer with optional momentum and weight decay.

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        momentum: Momentum factor (0 = no momentum).
        weight_decay: L2 regularization coefficient.
    """

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        """Perform one SGD update step."""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # Weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # Momentum
            if self.momentum != 0:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                update = self.velocities[i]
            else:
                update = grad

            # Parameter update
            param.data -= self.lr * update
