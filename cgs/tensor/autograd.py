"""
Reverse-mode Automatic Differentiation Engine.

Implements backpropagation through the computation graph built by
CGSTensor operations. Uses topological sorting to ensure correct
gradient flow ordering.
"""

import numpy as np
from typing import Optional


def backward(tensor, grad: Optional[np.ndarray] = None):
    """
    Perform reverse-mode automatic differentiation from a tensor.

    Traverses the computation graph in reverse topological order,
    calling each node's _backward_fn to propagate gradients.

    Args:
        tensor: The output tensor to differentiate from (typically the loss).
        grad: Initial gradient (defaults to 1.0 for scalar outputs).

    Raises:
        RuntimeError: If backward is called on a non-scalar without grad.
    """
    if grad is None:
        if tensor.data.size == 1:
            grad = np.ones_like(tensor.data)
        else:
            raise RuntimeError(
                "backward() requires grad argument for non-scalar tensors. "
                f"Got tensor with shape {tensor.shape}."
            )

    # Accumulate gradient on the output tensor
    if tensor.grad is None:
        tensor.grad = grad.copy()
    else:
        tensor.grad = tensor.grad + grad

    # Build topological ordering
    topo_order = _topological_sort(tensor)

    # Propagate gradients in reverse topological order
    for node in reversed(topo_order):
        if node._backward_fn is not None and node.grad is not None:
            node._backward_fn(node.grad)


def _topological_sort(tensor) -> list:
    """
    Topological sort of the computation graph rooted at tensor.

    Returns a list where each node appears after all its dependencies.
    Uses iterative DFS to avoid stack overflow on deep graphs.
    """
    visited = set()
    order = []

    # Iterative DFS using explicit stack
    stack = [(tensor, False)]

    while stack:
        node, processed = stack.pop()

        if processed:
            if id(node) not in visited:
                visited.add(id(node))
                order.append(node)
            continue

        if id(node) in visited:
            continue

        # Push this node back as "processed"
        stack.append((node, True))

        # Push children (parents in computation graph)
        for prev in node._prev:
            if id(prev) not in visited:
                stack.append((prev, False))

    return order


def numerical_gradient(fn, tensor, eps=1e-5):
    """
    Compute numerical gradient using central finite differences.

    Used for gradient checking / testing autograd correctness.

    Args:
        fn: Function that takes a CGSTensor and returns a scalar CGSTensor.
        tensor: The tensor to compute gradients for.
        eps: Perturbation size.

    Returns:
        np.ndarray: Numerical gradient with same shape as tensor.data.
    """
    grad = np.zeros_like(tensor.data)
    it = np.nditer(tensor.data, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old_val = tensor.data[idx]

        # f(x + eps)
        tensor.data[idx] = old_val + eps
        loss_plus = fn(tensor).data.item()

        # f(x - eps)
        tensor.data[idx] = old_val - eps
        loss_minus = fn(tensor).data.item()

        # Central difference
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)

        # Restore
        tensor.data[idx] = old_val
        it.iternext()

    return grad
