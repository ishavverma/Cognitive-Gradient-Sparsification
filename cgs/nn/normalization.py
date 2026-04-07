"""
Normalization Layers — LayerNorm and BatchNorm.
"""

import numpy as np
from .module import Module
from ..tensor.tensor import CGSTensor, is_grad_enabled
from ..tensor.ops import _should_track


class LayerNorm(Module):
    """
    Layer Normalization.

    Normalizes over the last dimension (features) for each sample independently.

    Args:
        normalized_shape: Size of the last dimension to normalize over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = CGSTensor(np.ones(normalized_shape), requires_grad=True, name='gamma')
        self.beta = CGSTensor(np.zeros(normalized_shape), requires_grad=True, name='beta')

    def forward(self, x: CGSTensor) -> CGSTensor:
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        out_data = self.gamma.data * x_norm + self.beta.data
        track = _should_track(x, self.gamma, self.beta)

        out = CGSTensor(out_data, requires_grad=track, _prev={x, self.gamma, self.beta})

        if track:
            def _backward(grad):
                N = self.normalized_shape
                std_inv = 1.0 / np.sqrt(var + self.eps)

                if self.gamma.requires_grad:
                    g_gamma = np.sum(grad * x_norm, axis=tuple(range(grad.ndim - 1)), keepdims=False)
                    if g_gamma.shape != self.gamma.shape:
                        g_gamma = g_gamma.reshape(self.gamma.shape)
                    self.gamma.grad = g_gamma if self.gamma.grad is None else self.gamma.grad + g_gamma

                if self.beta.requires_grad:
                    g_beta = np.sum(grad, axis=tuple(range(grad.ndim - 1)), keepdims=False)
                    if g_beta.shape != self.beta.shape:
                        g_beta = g_beta.reshape(self.beta.shape)
                    self.beta.grad = g_beta if self.beta.grad is None else self.beta.grad + g_beta

                if x.requires_grad:
                    dx_norm = grad * self.gamma.data
                    dvar = np.sum(dx_norm * (x.data - mean) * -0.5 * (var + self.eps) ** (-1.5),
                                 axis=-1, keepdims=True)
                    dmean = np.sum(dx_norm * -std_inv, axis=-1, keepdims=True) + \
                            dvar * np.mean(-2.0 * (x.data - mean), axis=-1, keepdims=True)
                    dx = dx_norm * std_inv + dvar * 2.0 * (x.data - mean) / N + dmean / N
                    x.grad = dx if x.grad is None else x.grad + dx

            out._backward_fn = _backward

        return out

    def __repr__(self):
        return f"LayerNorm({self.normalized_shape})"


class BatchNorm(Module):
    """
    Batch Normalization.

    Normalizes over the batch dimension. Maintains running statistics
    for eval mode.

    Args:
        num_features: Number of features (channels).
        eps: Numerical stability constant.
        momentum: Running statistics momentum.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = CGSTensor(np.ones(num_features), requires_grad=True, name='gamma')
        self.beta = CGSTensor(np.zeros(num_features), requires_grad=True, name='beta')

        # Running statistics (not trainable parameters)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: CGSTensor) -> CGSTensor:
        if self._training:
            # Compute batch statistics
            # x shape: (batch, features) or (batch, features, ...)
            axes = tuple(i for i in range(x.data.ndim) if i != 1) if x.data.ndim > 2 else (0,)
            mean = np.mean(x.data, axis=axes, keepdims=True)
            var = np.var(x.data, axis=axes, keepdims=True)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                                self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + \
                               self.momentum * var.squeeze()
        else:
            mean = self.running_mean.reshape(1, -1)
            var = self.running_var.reshape(1, -1)
            if x.data.ndim > 2:
                shape = [1, self.num_features] + [1] * (x.data.ndim - 2)
                mean = mean.reshape(shape)
                var = var.reshape(shape)

        x_norm = (x.data - mean) / np.sqrt(var + self.eps)

        # Reshape gamma/beta for broadcasting
        gamma = self.gamma.data
        beta = self.beta.data
        if x.data.ndim > 2:
            shape = [1, self.num_features] + [1] * (x.data.ndim - 2)
            gamma = gamma.reshape(shape)
            beta = beta.reshape(shape)
        else:
            gamma = gamma.reshape(1, -1)
            beta = beta.reshape(1, -1)

        out_data = gamma * x_norm + beta
        track = _should_track(x, self.gamma, self.beta)

        out = CGSTensor(out_data, requires_grad=track, _prev={x, self.gamma, self.beta})

        if track and self._training:
            batch_size = x.data.shape[0]

            def _backward(grad):
                std_inv = 1.0 / np.sqrt(var + self.eps)

                if self.gamma.requires_grad:
                    g_gamma = np.sum(grad * x_norm,
                                     axis=tuple(i for i in range(grad.ndim) if i != 1) if grad.ndim > 2 else (0,))
                    g_gamma = g_gamma.flatten()
                    self.gamma.grad = g_gamma if self.gamma.grad is None else self.gamma.grad + g_gamma

                if self.beta.requires_grad:
                    g_beta = np.sum(grad,
                                    axis=tuple(i for i in range(grad.ndim) if i != 1) if grad.ndim > 2 else (0,))
                    g_beta = g_beta.flatten()
                    self.beta.grad = g_beta if self.beta.grad is None else self.beta.grad + g_beta

                if x.requires_grad:
                    dx_norm = grad * gamma
                    N = batch_size
                    dvar = np.sum(dx_norm * (x.data - mean) * -0.5 * (var + self.eps) ** (-1.5),
                                 axis=0, keepdims=True)
                    dmean = np.sum(dx_norm * -std_inv, axis=0, keepdims=True)
                    dx = dx_norm * std_inv + dvar * 2.0 * (x.data - mean) / N + dmean / N
                    x.grad = dx if x.grad is None else x.grad + dx

            out._backward_fn = _backward

        return out

    def __repr__(self):
        return f"BatchNorm({self.num_features})"
