"""
Gradient Sparsifier — Threshold and Top-K gradient filtering.

The sparsifier takes GID scores and creates binary masks that
determine which gradients are applied. Supports:
  - Threshold mode: keep gradients where GID > τ
  - Top-K mode: keep top K% of gradients by GID
  - Hybrid mode: threshold + top-K combination
"""

import numpy as np
from typing import Dict, Tuple, Optional


class GradientSparsifier:
    """
    Filters gradients based on their GID scores.

    Only gradients that pass the filter are used for parameter updates.
    This is the mechanism that makes CGS training sparse.

    Args:
        mode: 'threshold', 'topk', or 'hybrid'.
        threshold: GID threshold for keeping gradients (threshold mode).
        top_k_ratio: Fraction of gradients to keep (topk mode).
        min_keep_ratio: Minimum fraction of gradients to always keep.
    """

    def __init__(self, mode: str = 'hybrid', threshold: float = 0.3,
                 top_k_ratio: float = 0.5, min_keep_ratio: float = 0.1):
        self.mode = mode
        self.threshold = threshold
        self.top_k_ratio = top_k_ratio
        self.min_keep_ratio = min_keep_ratio

        # Statistics
        self._step = 0
        self._sparsity_history: list = []

    def sparsify(self, gid_scores: Dict[str, float],
                 gradients: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Apply sparsification to gradients based on GID scores.

        Args:
            gid_scores: Dict of parameter_name → GID score.
            gradients: Dict of parameter_name → gradient array.

        Returns:
            Tuple of:
              - sparse_gradients: Dict with filtered gradients (zeros for filtered params)
              - stats: Sparsification statistics
        """
        self._step += 1

        if self.mode == 'threshold':
            mask = self._threshold_filter(gid_scores)
        elif self.mode == 'topk':
            mask = self._topk_filter(gid_scores)
        elif self.mode == 'hybrid':
            mask = self._hybrid_filter(gid_scores)
        else:
            raise ValueError(f"Unknown sparsification mode: {self.mode}")

        # Enforce minimum keep ratio
        mask = self._enforce_minimum(mask, gid_scores)

        # Apply mask to gradients
        sparse_gradients = {}
        kept = 0
        total = 0
        for name, grad in gradients.items():
            total += 1
            if mask.get(name, False):
                sparse_gradients[name] = grad
                kept += 1
            else:
                sparse_gradients[name] = np.zeros_like(grad)

        sparsity = 1.0 - (kept / max(total, 1))
        self._sparsity_history.append(sparsity)

        stats = {
            'total_params': total,
            'kept_params': kept,
            'filtered_params': total - kept,
            'sparsity': sparsity,
            'threshold_used': self.threshold,
        }

        return sparse_gradients, stats

    def _threshold_filter(self, scores: Dict[str, float]) -> Dict[str, bool]:
        """Keep gradients with GID ≥ threshold."""
        return {name: score >= self.threshold for name, score in scores.items()}

    def _topk_filter(self, scores: Dict[str, float]) -> Dict[str, bool]:
        """Keep top K% of gradients by GID score."""
        if not scores:
            return {}

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        k = max(1, int(len(sorted_items) * self.top_k_ratio))

        mask = {}
        for i, (name, score) in enumerate(sorted_items):
            mask[name] = i < k
        return mask

    def _hybrid_filter(self, scores: Dict[str, float]) -> Dict[str, bool]:
        """Combine threshold and top-K: must pass threshold AND be in top K%."""
        threshold_mask = self._threshold_filter(scores)
        topk_mask = self._topk_filter(scores)

        # Union: pass either filter (more permissive than intersection)
        mask = {}
        for name in scores:
            mask[name] = threshold_mask.get(name, False) or topk_mask.get(name, False)
        return mask

    def _enforce_minimum(self, mask: Dict[str, bool],
                        scores: Dict[str, float]) -> Dict[str, bool]:
        """Ensure at least min_keep_ratio of gradients are kept."""
        total = len(mask)
        kept = sum(1 for v in mask.values() if v)
        min_kept = max(1, int(total * self.min_keep_ratio))

        if kept < min_kept:
            # Activate the highest-scoring filtered gradients
            filtered = [(name, scores.get(name, 0))
                       for name, active in mask.items() if not active]
            filtered.sort(key=lambda x: x[1], reverse=True)

            for name, _ in filtered[:min_kept - kept]:
                mask[name] = True

        return mask

    def set_threshold(self, threshold: float):
        """Update sparsification threshold."""
        self.threshold = threshold

    def set_top_k_ratio(self, ratio: float):
        """Update top-K ratio."""
        self.top_k_ratio = np.clip(ratio, 0.05, 1.0)

    def get_stats(self) -> dict:
        """Get sparsification statistics."""
        if not self._sparsity_history:
            return {'avg_sparsity': 0, 'step': 0}

        recent = self._sparsity_history[-20:]
        return {
            'step': self._step,
            'avg_sparsity': float(np.mean(recent)),
            'current_sparsity': self._sparsity_history[-1] if self._sparsity_history else 0,
            'threshold': self.threshold,
            'top_k_ratio': self.top_k_ratio,
            'total_steps': len(self._sparsity_history),
        }
