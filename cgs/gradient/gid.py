"""
Gradient Information Density (GID) — A new metric to evaluate learning signals.

GID is the core metric of the CGS framework. It scores every gradient
based on three factors:
  1. Magnitude — strength of the learning signal
  2. Novelty — uniqueness compared to past gradients  
  3. Impact — predicted effect on model loss

GID = α·magnitude + β·novelty + γ·impact

Only gradients with high GID are used for parameter updates.
"""

import numpy as np
from typing import Dict, Optional, Tuple


class GradientInformationDensity:
    """
    Computes GID scores for gradients.

    GID measures how informative a gradient is for learning.
    High GID = the gradient carries valuable learning signal.
    Low GID = the gradient is redundant, noisy, or low-impact.

    Args:
        alpha: Weight for magnitude component.
        beta: Weight for novelty component.
        gamma: Weight for impact component.
        normalize: Whether to normalize each component to [0, 1].
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.4, gamma: float = 0.3,
                 normalize: bool = True):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.normalize = normalize

        # Running statistics for normalization
        self._magnitude_stats = _RunningStats()
        self._novelty_stats = _RunningStats()
        self._impact_stats = _RunningStats()

    def compute_gid(self, gradient_preview: Dict[str, dict],
                    memory_graph=None) -> Dict[str, float]:
        """
        Compute GID scores for all gradients.

        Args:
            gradient_preview: Output from GradientProbingLayer.probe().
            memory_graph: GradientMemoryGraph for novelty computation.

        Returns:
            Dict mapping parameter names to GID scores.
        """
        gid_scores = {}

        for name, info in gradient_preview.items():
            grad = info['gradient']

            # 1. Magnitude score
            magnitude = self._compute_magnitude(grad)

            # 2. Novelty score  
            if memory_graph is not None:
                novelty = memory_graph.query_novelty(name, grad)
            else:
                novelty = 1.0  # If no memory, everything is novel

            # 3. Impact score
            impact = abs(info.get('predicted_loss_change', 0.0))

            # Update running statistics
            self._magnitude_stats.update(magnitude)
            self._novelty_stats.update(novelty)
            self._impact_stats.update(impact)

            # Normalize to [0, 1] using running statistics
            if self.normalize:
                magnitude = self._magnitude_stats.normalize(magnitude)
                novelty = self._novelty_stats.normalize(novelty)
                impact = self._impact_stats.normalize(impact)

            # Weighted combination
            gid = self.alpha * magnitude + self.beta * novelty + self.gamma * impact

            # Bonus: direction stability adds confidence
            stability = info.get('direction_stability', 1.0)
            gid *= (0.5 + 0.5 * stability)  # Scale by 50-100%

            gid_scores[name] = float(gid)

        return gid_scores

    def compute_gid_simple(self, gradients: Dict[str, np.ndarray],
                           memory_graph=None) -> Dict[str, float]:
        """
        Simplified GID computation from raw gradients (no probing info).

        Used with quick_probe() or standard backward pass.
        """
        gid_scores = {}

        for name, grad in gradients.items():
            magnitude = self._compute_magnitude(grad)

            if memory_graph is not None:
                novelty = memory_graph.query_novelty(name, grad)
            else:
                novelty = 1.0

            # No impact score without probing
            impact = magnitude  # Use magnitude as proxy

            self._magnitude_stats.update(magnitude)
            self._novelty_stats.update(novelty)
            self._impact_stats.update(impact)

            if self.normalize:
                magnitude = self._magnitude_stats.normalize(magnitude)
                novelty = self._novelty_stats.normalize(novelty)
                impact = self._impact_stats.normalize(impact)

            gid = self.alpha * magnitude + self.beta * novelty + self.gamma * impact
            gid_scores[name] = float(gid)

        return gid_scores

    @staticmethod
    def _compute_magnitude(grad: np.ndarray) -> float:
        """Compute gradient magnitude (L2 norm)."""
        return float(np.sqrt(np.sum(grad ** 2)))

    def get_weights(self) -> Tuple[float, float, float]:
        """Get current GID component weights."""
        return self.alpha, self.beta, self.gamma

    def set_weights(self, alpha: float, beta: float, gamma: float):
        """Update GID component weights."""
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def reset_stats(self):
        """Reset running normalization statistics."""
        self._magnitude_stats = _RunningStats()
        self._novelty_stats = _RunningStats()
        self._impact_stats = _RunningStats()


class _RunningStats:
    """
    Running mean/std tracker for online normalization.

    Uses Welford's algorithm for numerical stability.
    """

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def update(self, x: float):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return np.sqrt(self.M2 / (self.count - 1))

    def normalize(self, x: float) -> float:
        """Normalize to approximately [0, 1] using min-max scaling."""
        if self.count < 2 or self.max_val <= self.min_val:
            return 0.5
        return np.clip((x - self.min_val) / (self.max_val - self.min_val + 1e-8), 0.0, 1.0)
