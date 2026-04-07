"""
Adam Optimizer — Adaptive Moment Estimation.
"""

import numpy as np
from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer with first/second moment estimation and bias correction.

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        beta1: Exponential decay rate for first moment (mean).
        beta2: Exponential decay rate for second moment (variance).
        eps: Small constant for numerical stability.
        weight_decay: L2 regularization coefficient.
    """

    def __init__(self, params, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        # First moment (mean) and second moment (variance) estimates
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0  # Timestep

    def step(self):
        """Perform one Adam update step."""
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # Weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Parameter update
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
