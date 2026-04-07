"""
Loss Function Modules.

Module wrappers around the functional loss implementations.
"""

from .module import Module
from ..tensor.tensor import CGSTensor
from ..tensor import functional as F


class CrossEntropyLoss(Module):
    """
    Cross-entropy loss with built-in log-softmax.

    Expects raw logits (not softmax probabilities) and integer targets.

    Args:
        logits: (batch, num_classes) raw model outputs.
        targets: (batch,) integer class labels.

    Returns:
        Scalar loss.
    """

    def forward(self, logits: CGSTensor, targets: CGSTensor) -> CGSTensor:
        return F.cross_entropy(logits, targets)

    def __repr__(self):
        return "CrossEntropyLoss()"


class MSELoss(Module):
    """
    Mean Squared Error loss.

    MSE = mean((pred - target)^2)
    """

    def forward(self, pred: CGSTensor, target: CGSTensor) -> CGSTensor:
        return F.mse_loss(pred, target)

    def __repr__(self):
        return "MSELoss()"


class ContrastiveLoss(Module):
    """
    NT-Xent Contrastive Loss for self-supervised multi-view learning.

    Args:
        temperature: Temperature scaling parameter.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: CGSTensor, z2: CGSTensor) -> CGSTensor:
        return F.contrastive_loss(z1, z2, self.temperature)

    def __repr__(self):
        return f"ContrastiveLoss(temperature={self.temperature})"
