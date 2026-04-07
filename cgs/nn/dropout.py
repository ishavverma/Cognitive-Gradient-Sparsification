"""
Dropout Layer.

Randomly zeros elements during training to prevent overfitting.
"""

import numpy as np
from .module import Module
from ..tensor.tensor import CGSTensor
from ..tensor.ops import _should_track


class Dropout(Module):
    """
    Dropout regularization layer.

    During training, randomly zeroes elements with probability p.
    During evaluation, passes input unchanged.

    Args:
        p: Dropout probability (fraction of elements to zero). Default: 0.5.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: CGSTensor) -> CGSTensor:
        if not self._training or self.p == 0:
            return x

        # Generate dropout mask (inverted dropout — scale by 1/(1-p))
        mask = (np.random.rand(*x.shape) > self.p).astype(x.data.dtype)
        scale = 1.0 / (1.0 - self.p)
        out_data = x.data * mask * scale
        track = _should_track(x)

        out = CGSTensor(out_data, requires_grad=track, _prev={x})

        if track:
            def _backward(grad):
                if x.requires_grad:
                    x_grad = grad * mask * scale
                    x.grad = x_grad if x.grad is None else x.grad + x_grad
            out._backward_fn = _backward

        return out

    def __repr__(self):
        return f"Dropout(p={self.p})"
