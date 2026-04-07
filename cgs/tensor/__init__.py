"""CGS Tensor Engine — Custom tensor with automatic differentiation."""

from .tensor import CGSTensor, no_grad
from .ops import matmul, add, relu, sigmoid, tanh, tensor_sum, tensor_mean, reshape, transpose
from .functional import softmax, log_softmax, cross_entropy, mse_loss, contrastive_loss, cosine_similarity

__all__ = [
    'CGSTensor', 'no_grad',
    'matmul', 'add', 'relu', 'sigmoid', 'tanh', 'tensor_sum', 'tensor_mean',
    'reshape', 'transpose',
    'softmax', 'log_softmax', 'cross_entropy', 'mse_loss',
    'contrastive_loss', 'cosine_similarity',
]
