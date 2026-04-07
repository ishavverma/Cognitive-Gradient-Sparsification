"""
CGSTensor: Core tensor class with automatic differentiation support.

This is the foundational building block of the CGS framework. Every
computation flows through CGSTensor objects, which track their computation
graph for reverse-mode automatic differentiation.
"""

import numpy as np
from typing import Optional, Tuple, Set, Callable, Union

# Global flag to disable gradient tracking
_grad_enabled = True


class _NoGradContext:
    """Context manager to disable gradient tracking."""

    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self._prev


def no_grad():
    """Context manager that disables gradient computation.

    Usage:
        with no_grad():
            output = model(input)  # No gradients tracked
    """
    return _NoGradContext()


def is_grad_enabled() -> bool:
    """Check if gradient tracking is currently enabled."""
    return _grad_enabled


class CGSTensor:
    """
    A tensor with automatic differentiation support.

    Wraps a NumPy ndarray and tracks the computation graph for
    reverse-mode autodiff (backpropagation).

    Attributes:
        data (np.ndarray): The underlying numerical data.
        grad (np.ndarray or None): Accumulated gradient after backward().
        requires_grad (bool): Whether to track gradients for this tensor.
        _prev (set): Parent tensors in the computation graph.
        _backward_fn (callable): Function to propagate gradients backward.
        _name (str): Optional name for debugging.
    """

    def __init__(
        self,
        data: Union[np.ndarray, list, float, int],
        requires_grad: bool = False,
        _prev: Optional[Set] = None,
        _backward_fn: Optional[Callable] = None,
        name: str = "",
    ):
        if isinstance(data, CGSTensor):
            data = data.data
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)
        elif data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float64)

        self.data = data
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._prev: Set['CGSTensor'] = _prev or set()
        self._backward_fn: Optional[Callable] = _backward_fn
        self._name = name

    # ──────────────────── Properties ────────────────────

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> 'CGSTensor':
        """Transpose shorthand."""
        from .ops import transpose
        return transpose(self)

    # ──────────────────── Gradient Ops ────────────────────

    def zero_grad(self):
        """Reset the gradient to None."""
        self.grad = None

    def backward(self, grad: Optional[np.ndarray] = None):
        """
        Perform reverse-mode automatic differentiation.

        Computes gradients for all tensors in the computation graph
        that have requires_grad=True.

        Args:
            grad: Upstream gradient. Defaults to ones (for scalar loss).
        """
        from .autograd import backward
        backward(self, grad)

    def detach(self) -> 'CGSTensor':
        """Return a new tensor detached from the computation graph."""
        return CGSTensor(self.data.copy(), requires_grad=False, name=self._name + "_detached")

    # ──────────────────── Operator Overloads ────────────────────

    def __add__(self, other):
        from .ops import add
        other = _ensure_tensor(other)
        return add(self, other)

    def __radd__(self, other):
        from .ops import add
        other = _ensure_tensor(other)
        return add(other, self)

    def __sub__(self, other):
        from .ops import sub
        other = _ensure_tensor(other)
        return sub(self, other)

    def __rsub__(self, other):
        from .ops import sub
        other = _ensure_tensor(other)
        return sub(other, self)

    def __mul__(self, other):
        from .ops import mul
        other = _ensure_tensor(other)
        return mul(self, other)

    def __rmul__(self, other):
        from .ops import mul
        other = _ensure_tensor(other)
        return mul(other, self)

    def __truediv__(self, other):
        from .ops import div
        other = _ensure_tensor(other)
        return div(self, other)

    def __rtruediv__(self, other):
        from .ops import div
        other = _ensure_tensor(other)
        return div(other, self)

    def __matmul__(self, other):
        from .ops import matmul
        other = _ensure_tensor(other)
        return matmul(self, other)

    def __neg__(self):
        from .ops import neg
        return neg(self)

    def __pow__(self, power):
        from .ops import power as pow_op
        return pow_op(self, power)

    # ──────────────────── Reductions ────────────────────

    def sum(self, axis=None, keepdims=False):
        from .ops import tensor_sum
        return tensor_sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        from .ops import tensor_mean
        return tensor_mean(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        from .ops import tensor_max
        return tensor_max(self, axis=axis, keepdims=keepdims)

    # ──────────────────── Shape Ops ────────────────────

    def reshape(self, *shape):
        from .ops import reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape(self, shape)

    def flatten(self):
        return self.reshape(-1)

    # ──────────────────── Indexing ────────────────────

    def __getitem__(self, key):
        from .ops import tensor_slice
        return tensor_slice(self, key)

    # ──────────────────── Representation ────────────────────

    def __repr__(self):
        grad_info = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        name_info = f", name='{self._name}'" if self._name else ""
        return f"CGSTensor({self.data}{grad_info}{name_info})"

    def __len__(self):
        return self.shape[0]

    def numpy(self) -> np.ndarray:
        """Return the underlying NumPy array (detached)."""
        return self.data.copy()


def _ensure_tensor(x) -> CGSTensor:
    """Convert scalars and arrays to CGSTensor."""
    if isinstance(x, CGSTensor):
        return x
    return CGSTensor(x)
