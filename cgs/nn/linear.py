"""
Linear (Dense / Fully-Connected) Layer.

Implements y = xW^T + b with Xavier/He initialization.
"""

import numpy as np
from .module import Module
from ..tensor.tensor import CGSTensor
from ..tensor.ops import matmul, add


class Linear(Module):
    """
    Fully-connected linear layer.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to include a bias term.
        init: Weight initialization strategy ('xavier', 'he', 'normal').
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, init: str = 'xavier'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight initialization
        if init == 'xavier':
            std = np.sqrt(2.0 / (in_features + out_features))
        elif init == 'he':
            std = np.sqrt(2.0 / in_features)
        else:
            std = 0.01

        w_data = np.random.randn(out_features, in_features) * std
        self.weight = CGSTensor(w_data, requires_grad=True, name='weight')

        if bias:
            b_data = np.zeros(out_features)
            self.bias = CGSTensor(b_data, requires_grad=True, name='bias')
        else:
            self.has_bias = False

    def forward(self, x: CGSTensor) -> CGSTensor:
        """
        Forward pass: y = xW^T + b

        Args:
            x: Input tensor, shape (..., in_features).

        Returns:
            Output tensor, shape (..., out_features).
        """
        # x @ W^T
        from ..tensor.ops import transpose
        out = matmul(x, transpose(self.weight))

        if 'bias' in self._parameters:
            out = add(out, self.bias)

        return out

    def __repr__(self):
        bias_str = 'bias' in self._parameters
        return f"Linear(in={self.in_features}, out={self.out_features}, bias={bias_str})"
