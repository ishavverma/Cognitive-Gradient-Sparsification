"""
Training Metrics — Accuracy, Efficiency, Sparsity tracking.
"""

import numpy as np
from typing import List, Dict


class Accuracy:
    """Accuracy metric tracker."""

    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)
        targets = targets.astype(int)
        self.correct += int(np.sum(predictions == targets))
        self.total += len(targets)

    def compute(self) -> float:
        return self.correct / max(self.total, 1)

    def reset(self):
        self.correct = 0
        self.total = 0


class TrainingEfficiency:
    """Track training efficiency metrics."""

    def __init__(self):
        self.total_updates = 0
        self.useful_updates = 0  # Updates that decreased loss
        self.total_params_updated = 0
        self.total_params_available = 0

    def update(self, params_updated: int, params_available: int, loss_decreased: bool):
        self.total_updates += 1
        self.total_params_updated += params_updated
        self.total_params_available += params_available
        if loss_decreased:
            self.useful_updates += 1

    def compute(self) -> dict:
        return {
            'update_efficiency': self.useful_updates / max(self.total_updates, 1),
            'param_efficiency': self.total_params_updated / max(self.total_params_available, 1),
            'total_updates': self.total_updates,
            'useful_updates': self.useful_updates,
        }

    def reset(self):
        self.total_updates = 0
        self.useful_updates = 0
        self.total_params_updated = 0
        self.total_params_available = 0


class SparsityTracker:
    """Track gradient sparsity over training."""

    def __init__(self):
        self.history: List[float] = []

    def update(self, sparsity: float):
        self.history.append(sparsity)

    def compute(self) -> dict:
        if not self.history:
            return {'avg_sparsity': 0, 'current_sparsity': 0}
        return {
            'avg_sparsity': float(np.mean(self.history)),
            'current_sparsity': self.history[-1],
            'max_sparsity': float(np.max(self.history)),
            'min_sparsity': float(np.min(self.history)),
        }

    def reset(self):
        self.history.clear()
