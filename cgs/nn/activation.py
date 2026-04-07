"""
Activation Function Modules.

Thin Module wrappers around the primitive activation ops.
"""

from .module import Module
from ..tensor.tensor import CGSTensor
from ..tensor import ops


class ReLU(Module):
    """Rectified Linear Unit activation."""

    def forward(self, x: CGSTensor) -> CGSTensor:
        return ops.relu(x)

    def __repr__(self):
        return "ReLU()"


class GELU(Module):
    """Gaussian Error Linear Unit activation (approximate)."""

    def forward(self, x: CGSTensor) -> CGSTensor:
        return ops.gelu(x)

    def __repr__(self):
        return "GELU()"


class Sigmoid(Module):
    """Sigmoid activation."""

    def forward(self, x: CGSTensor) -> CGSTensor:
        return ops.sigmoid(x)

    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    """Hyperbolic tangent activation."""

    def forward(self, x: CGSTensor) -> CGSTensor:
        return ops.tanh(x)

    def __repr__(self):
        return "Tanh()"
