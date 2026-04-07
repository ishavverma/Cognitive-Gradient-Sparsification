"""
Primitive Differentiable Operations.

Each operation creates a new CGSTensor and registers a backward function
that computes gradients with respect to its inputs. These form the
atomic building blocks of the computation graph.
"""

import numpy as np
from .tensor import CGSTensor, _ensure_tensor, is_grad_enabled


def _should_track(*tensors) -> bool:
    """Check if any input requires gradient and tracking is enabled."""
    return is_grad_enabled() and any(t.requires_grad for t in tensors)


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Sum out broadcasted dimensions to match original shape.

    When a (3,) tensor is added to a (4, 3) tensor, the gradient of
    the (3,) tensor must be summed over axis 0 to get back to (3,).
    """
    # Handle scalar case
    if shape == ():
        return np.sum(grad).reshape(())

    # Add leading dimensions if needed
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # Sum over broadcasted axes
    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


# ──────────────────── Arithmetic Ops ────────────────────


def add(a: CGSTensor, b: CGSTensor) -> CGSTensor:
    """Element-wise addition with gradient tracking."""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    out_data = a.data + b.data
    track = _should_track(a, b)

    out = CGSTensor(out_data, requires_grad=track, _prev={a, b})

    if track:
        def _backward(grad):
            if a.requires_grad:
                a_grad = _unbroadcast(grad, a.shape)
                a.grad = a_grad if a.grad is None else a.grad + a_grad
            if b.requires_grad:
                b_grad = _unbroadcast(grad, b.shape)
                b.grad = b_grad if b.grad is None else b.grad + b_grad
        out._backward_fn = _backward

    return out


def sub(a: CGSTensor, b: CGSTensor) -> CGSTensor:
    """Element-wise subtraction with gradient tracking."""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    out_data = a.data - b.data
    track = _should_track(a, b)

    out = CGSTensor(out_data, requires_grad=track, _prev={a, b})

    if track:
        def _backward(grad):
            if a.requires_grad:
                a_grad = _unbroadcast(grad, a.shape)
                a.grad = a_grad if a.grad is None else a.grad + a_grad
            if b.requires_grad:
                b_grad = _unbroadcast(-grad, b.shape)
                b.grad = b_grad if b.grad is None else b.grad + b_grad
        out._backward_fn = _backward

    return out


def mul(a: CGSTensor, b: CGSTensor) -> CGSTensor:
    """Element-wise multiplication with gradient tracking."""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    out_data = a.data * b.data
    track = _should_track(a, b)

    out = CGSTensor(out_data, requires_grad=track, _prev={a, b})

    if track:
        def _backward(grad):
            if a.requires_grad:
                a_grad = _unbroadcast(grad * b.data, a.shape)
                a.grad = a_grad if a.grad is None else a.grad + a_grad
            if b.requires_grad:
                b_grad = _unbroadcast(grad * a.data, b.shape)
                b.grad = b_grad if b.grad is None else b.grad + b_grad
        out._backward_fn = _backward

    return out


def div(a: CGSTensor, b: CGSTensor) -> CGSTensor:
    """Element-wise division with gradient tracking."""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    out_data = a.data / b.data
    track = _should_track(a, b)

    out = CGSTensor(out_data, requires_grad=track, _prev={a, b})

    if track:
        def _backward(grad):
            if a.requires_grad:
                a_grad = _unbroadcast(grad / b.data, a.shape)
                a.grad = a_grad if a.grad is None else a.grad + a_grad
            if b.requires_grad:
                b_grad = _unbroadcast(-grad * a.data / (b.data ** 2), b.shape)
                b.grad = b_grad if b.grad is None else b.grad + b_grad
        out._backward_fn = _backward

    return out


def neg(a: CGSTensor) -> CGSTensor:
    """Negation with gradient tracking."""
    out_data = -a.data
    track = _should_track(a)

    out = CGSTensor(out_data, requires_grad=track, _prev={a})

    if track:
        def _backward(grad):
            if a.requires_grad:
                a.grad = -grad if a.grad is None else a.grad - grad
        out._backward_fn = _backward

    return out


def power(a: CGSTensor, p: float) -> CGSTensor:
    """Element-wise power with gradient tracking."""
    out_data = a.data ** p
    track = _should_track(a)

    out = CGSTensor(out_data, requires_grad=track, _prev={a})

    if track:
        def _backward(grad):
            if a.requires_grad:
                a_grad = grad * p * (a.data ** (p - 1))
                a.grad = a_grad if a.grad is None else a.grad + a_grad
        out._backward_fn = _backward

    return out


# ──────────────────── Matrix Ops ────────────────────


def matmul(a: CGSTensor, b: CGSTensor) -> CGSTensor:
    """Matrix multiplication with gradient tracking.

    Supports:
      - 2D @ 2D → standard matmul
      - batched matmul (3D @ 2D or 3D @ 3D)
    """
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    out_data = a.data @ b.data
    track = _should_track(a, b)

    out = CGSTensor(out_data, requires_grad=track, _prev={a, b})

    if track:
        def _backward(grad):
            if a.requires_grad:
                if b.data.ndim == 1:
                    a_grad = np.outer(grad, b.data) if grad.ndim == 1 else grad[..., None] * b.data[None, :]
                else:
                    a_grad = grad @ np.swapaxes(b.data, -1, -2)
                a_grad = _unbroadcast(a_grad, a.shape)
                a.grad = a_grad if a.grad is None else a.grad + a_grad
            if b.requires_grad:
                if a.data.ndim == 1:
                    b_grad = np.outer(a.data, grad) if grad.ndim == 1 else a.data[:, None] * grad[None, :]
                else:
                    b_grad = np.swapaxes(a.data, -1, -2) @ grad
                b_grad = _unbroadcast(b_grad, b.shape)
                b.grad = b_grad if b.grad is None else b.grad + b_grad
        out._backward_fn = _backward

    return out


def transpose(a: CGSTensor, axes=None) -> CGSTensor:
    """Transpose with gradient tracking."""
    if axes is None:
        out_data = a.data.T
    else:
        out_data = np.transpose(a.data, axes)
    track = _should_track(a)

    out = CGSTensor(out_data, requires_grad=track, _prev={a})

    if track:
        def _backward(grad):
            if a.requires_grad:
                if axes is None:
                    a_grad = grad.T
                else:
                    # Invert the permutation
                    inv_axes = [0] * len(axes)
                    for i, ax in enumerate(axes):
                        inv_axes[ax] = i
                    a_grad = np.transpose(grad, inv_axes)
                a.grad = a_grad if a.grad is None else a.grad + a_grad
        out._backward_fn = _backward

    return out


# ──────────────────── Activation Ops ────────────────────


def relu(x: CGSTensor) -> CGSTensor:
    """ReLU activation with gradient tracking."""
    out_data = np.maximum(0, x.data)
    track = _should_track(x)

    out = CGSTensor(out_data, requires_grad=track, _prev={x})

    if track:
        def _backward(grad):
            if x.requires_grad:
                x_grad = grad * (x.data > 0).astype(grad.dtype)
                x.grad = x_grad if x.grad is None else x.grad + x_grad
        out._backward_fn = _backward

    return out


def sigmoid(x: CGSTensor) -> CGSTensor:
    """Sigmoid activation with gradient tracking."""
    s = 1.0 / (1.0 + np.exp(-np.clip(x.data, -500, 500)))
    track = _should_track(x)

    out = CGSTensor(s, requires_grad=track, _prev={x})

    if track:
        def _backward(grad):
            if x.requires_grad:
                x_grad = grad * s * (1 - s)
                x.grad = x_grad if x.grad is None else x.grad + x_grad
        out._backward_fn = _backward

    return out


def tanh(x: CGSTensor) -> CGSTensor:
    """Tanh activation with gradient tracking."""
    t = np.tanh(x.data)
    track = _should_track(x)

    out = CGSTensor(t, requires_grad=track, _prev={x})

    if track:
        def _backward(grad):
            if x.requires_grad:
                x_grad = grad * (1 - t ** 2)
                x.grad = x_grad if x.grad is None else x.grad + x_grad
        out._backward_fn = _backward

    return out


def gelu(x: CGSTensor) -> CGSTensor:
    """GELU activation (approximate) with gradient tracking."""
    # Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x.data + 0.044715 * x.data ** 3)
    t = np.tanh(inner)
    out_data = 0.5 * x.data * (1 + t)
    track = _should_track(x)

    out = CGSTensor(out_data, requires_grad=track, _prev={x})

    if track:
        def _backward(grad):
            if x.requires_grad:
                sech2 = 1 - t ** 2
                d_inner = c * (1 + 3 * 0.044715 * x.data ** 2)
                x_grad = grad * (0.5 * (1 + t) + 0.5 * x.data * sech2 * d_inner)
                x.grad = x_grad if x.grad is None else x.grad + x_grad
        out._backward_fn = _backward

    return out


# ──────────────────── Reduction Ops ────────────────────


def tensor_sum(x: CGSTensor, axis=None, keepdims=False) -> CGSTensor:
    """Sum reduction with gradient tracking."""
    out_data = np.sum(x.data, axis=axis, keepdims=keepdims)
    track = _should_track(x)

    out = CGSTensor(out_data, requires_grad=track, _prev={x})

    if track:
        def _backward(grad):
            if x.requires_grad:
                # Expand grad back to input shape
                if axis is not None and not keepdims:
                    expanded_shape = list(x.shape)
                    if isinstance(axis, int):
                        axes = [axis]
                    else:
                        axes = list(axis)
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, axis=ax)
                x_grad = np.broadcast_to(grad, x.shape).copy()
                x.grad = x_grad if x.grad is None else x.grad + x_grad
        out._backward_fn = _backward

    return out


def tensor_mean(x: CGSTensor, axis=None, keepdims=False) -> CGSTensor:
    """Mean reduction with gradient tracking."""
    out_data = np.mean(x.data, axis=axis, keepdims=keepdims)
    track = _should_track(x)

    # Compute count of elements being averaged
    if axis is None:
        count = x.data.size
    elif isinstance(axis, int):
        count = x.data.shape[axis]
    else:
        count = 1
        for ax in axis:
            count *= x.data.shape[ax]

    out = CGSTensor(out_data, requires_grad=track, _prev={x})

    if track:
        def _backward(grad):
            if x.requires_grad:
                if axis is not None and not keepdims:
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                x_grad = np.broadcast_to(grad / count, x.shape).copy()
                x.grad = x_grad if x.grad is None else x.grad + x_grad
        out._backward_fn = _backward

    return out


def tensor_max(x: CGSTensor, axis=None, keepdims=False) -> CGSTensor:
    """Max reduction with gradient tracking (straight-through for ties)."""
    out_data = np.max(x.data, axis=axis, keepdims=keepdims)
    track = _should_track(x)

    out = CGSTensor(out_data, requires_grad=track, _prev={x})

    if track:
        def _backward(grad):
            if x.requires_grad:
                if axis is not None and not keepdims:
                    expanded = np.expand_dims(out_data, axis=axis)
                    expanded_grad = np.expand_dims(grad, axis=axis)
                else:
                    expanded = out_data
                    expanded_grad = grad
                mask = (x.data == np.broadcast_to(expanded, x.shape)).astype(grad.dtype)
                # Normalize mask for ties
                mask_sum = np.sum(mask, axis=axis, keepdims=True)
                mask_sum = np.maximum(mask_sum, 1.0)
                mask = mask / mask_sum
                x_grad = mask * np.broadcast_to(expanded_grad, x.shape)
                x.grad = x_grad if x.grad is None else x.grad + x_grad
        out._backward_fn = _backward

    return out


# ──────────────────── Shape Ops ────────────────────


def reshape(x: CGSTensor, shape: tuple) -> CGSTensor:
    """Reshape with gradient tracking."""
    out_data = x.data.reshape(shape)
    track = _should_track(x)

    out = CGSTensor(out_data, requires_grad=track, _prev={x})

    if track:
        def _backward(grad):
            if x.requires_grad:
                x_grad = grad.reshape(x.shape)
                x.grad = x_grad if x.grad is None else x.grad + x_grad
        out._backward_fn = _backward

    return out


def tensor_slice(x: CGSTensor, key) -> CGSTensor:
    """Indexing / slicing with gradient tracking."""
    out_data = x.data[key]
    track = _should_track(x)

    out = CGSTensor(out_data, requires_grad=track, _prev={x})

    if track:
        def _backward(grad):
            if x.requires_grad:
                x_grad = np.zeros_like(x.data)
                x_grad[key] = grad
                x.grad = x_grad if x.grad is None else x.grad + x_grad
        out._backward_fn = _backward

    return out


# ──────────────────── Other Ops ────────────────────


def exp(x: CGSTensor) -> CGSTensor:
    """Element-wise exp with gradient tracking."""
    out_data = np.exp(np.clip(x.data, -500, 500))
    track = _should_track(x)

    out = CGSTensor(out_data, requires_grad=track, _prev={x})

    if track:
        def _backward(grad):
            if x.requires_grad:
                x_grad = grad * out_data
                x.grad = x_grad if x.grad is None else x.grad + x_grad
        out._backward_fn = _backward

    return out


def log(x: CGSTensor) -> CGSTensor:
    """Element-wise natural log with gradient tracking."""
    out_data = np.log(np.clip(x.data, 1e-12, None))
    track = _should_track(x)

    out = CGSTensor(out_data, requires_grad=track, _prev={x})

    if track:
        def _backward(grad):
            if x.requires_grad:
                x_grad = grad / np.clip(x.data, 1e-12, None)
                x.grad = x_grad if x.grad is None else x.grad + x_grad
        out._backward_fn = _backward

    return out


def concatenate(tensors: list, axis: int = 0) -> CGSTensor:
    """Concatenate tensors along an axis with gradient tracking."""
    data_list = [t.data for t in tensors]
    out_data = np.concatenate(data_list, axis=axis)
    track = any(_should_track(t) for t in tensors)

    out = CGSTensor(out_data, requires_grad=track, _prev=set(tensors))

    if track:
        def _backward(grad):
            # Split gradient along concat axis
            sizes = [t.shape[axis] for t in tensors]
            splits = np.cumsum(sizes)[:-1]
            grads = np.split(grad, splits, axis=axis)
            for t, g in zip(tensors, grads):
                if t.requires_grad:
                    t.grad = g if t.grad is None else t.grad + g
        out._backward_fn = _backward

    return out
