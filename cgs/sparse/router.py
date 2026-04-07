"""
Sparse Update Router (SUR) — Dynamic Learning Pathways.

Instead of updating all layers uniformly, SUR decides:
  - Which layers to update
  - Which neurons to activate
  - Which gradients to ignore

Creates dynamic learning pathways that change during training.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .parameter_block import ModularParameterBlock, BlockState


class SparseUpdateRouter:
    """
    Routes gradient updates to specific parameter blocks based on importance.

    Generates a routing matrix:
      Layer_i ← Activated if Importance_i > τ

    This creates dynamic learning pathways — different parts of the
    model learn at different rates depending on gradient quality.

    Args:
        threshold: Base importance threshold for activation.
        min_active_ratio: Minimum fraction of blocks that must be active.
        max_active_ratio: Maximum fraction of blocks that can be active.
        routing_mode: 'threshold', 'topk', or 'adaptive'.
    """

    def __init__(self, threshold: float = 0.3, min_active_ratio: float = 0.2,
                 max_active_ratio: float = 0.9, routing_mode: str = 'adaptive'):
        self.threshold = threshold
        self.min_active_ratio = min_active_ratio
        self.max_active_ratio = max_active_ratio
        self.routing_mode = routing_mode

        # Statistics
        self._step = 0
        self._routing_history: List[dict] = []

    def route(self, importance_map: Dict[str, float],
              parameter_blocks: List[ModularParameterBlock]) -> Dict[str, bool]:
        """
        Decide which parameter blocks to update.

        Args:
            importance_map: GID scores per parameter name.
            parameter_blocks: List of ModularParameterBlocks.

        Returns:
            routing_decisions: Dict mapping block names to update (True/False).
        """
        self._step += 1

        if self.routing_mode == 'threshold':
            decisions = self._threshold_routing(importance_map, parameter_blocks)
        elif self.routing_mode == 'topk':
            decisions = self._topk_routing(importance_map, parameter_blocks)
        elif self.routing_mode == 'adaptive':
            decisions = self._adaptive_routing(importance_map, parameter_blocks)
        else:
            raise ValueError(f"Unknown routing mode: {self.routing_mode}")

        # Apply decisions to blocks
        for block in parameter_blocks:
            if decisions.get(block.name, False):
                block.activate()
            else:
                block.freeze()

        # Track routing statistics
        active_count = sum(1 for v in decisions.values() if v)
        total_count = len(decisions)
        self._routing_history.append({
            'step': self._step,
            'active_ratio': active_count / max(total_count, 1),
            'active_blocks': [n for n, v in decisions.items() if v],
            'frozen_blocks': [n for n, v in decisions.items() if not v],
        })

        return decisions

    def _threshold_routing(self, importance_map: Dict[str, float],
                          blocks: List[ModularParameterBlock]) -> Dict[str, bool]:
        """Simple threshold-based routing."""
        decisions = {}
        for block in blocks:
            # Average importance of parameters in this block
            block_importance = self._compute_block_importance(block, importance_map)
            decisions[block.name] = block_importance >= self.threshold
        return self._enforce_constraints(decisions)

    def _topk_routing(self, importance_map: Dict[str, float],
                     blocks: List[ModularParameterBlock]) -> Dict[str, bool]:
        """Top-K routing: activate K most important blocks."""
        block_scores = []
        for block in blocks:
            score = self._compute_block_importance(block, importance_map)
            block_scores.append((block.name, score))

        block_scores.sort(key=lambda x: x[1], reverse=True)
        k = max(1, int(len(blocks) * self.max_active_ratio))

        decisions = {}
        for i, (name, score) in enumerate(block_scores):
            decisions[name] = i < k
        return decisions

    def _adaptive_routing(self, importance_map: Dict[str, float],
                         blocks: List[ModularParameterBlock]) -> Dict[str, bool]:
        """
        Adaptive routing: dynamically adjust based on importance distribution.

        If importances are spread out → activate more blocks (model needs broad learning)
        If importances are concentrated → activate fewer blocks (focus learning)
        """
        block_scores = {}
        all_importances = []

        for block in blocks:
            score = self._compute_block_importance(block, importance_map)
            block_scores[block.name] = score
            all_importances.append(score)

        if not all_importances:
            return {b.name: True for b in blocks}

        # Compute adaptive threshold based on distribution
        mean_imp = np.mean(all_importances)
        std_imp = np.std(all_importances)

        # High std → concentrated → use higher threshold
        # Low std → spread → use lower threshold
        if std_imp > 0.1:
            adaptive_threshold = mean_imp + 0.5 * std_imp
        else:
            adaptive_threshold = mean_imp - 0.5 * std_imp

        adaptive_threshold = max(adaptive_threshold, self.threshold * 0.5)

        decisions = {}
        for block in blocks:
            decisions[block.name] = block_scores.get(block.name, 0) >= adaptive_threshold

        return self._enforce_constraints(decisions)

    def _compute_block_importance(self, block: ModularParameterBlock,
                                  importance_map: Dict[str, float]) -> float:
        """Compute average importance of parameters in a block."""
        importances = []
        for param_name in block.parameter_names:
            if param_name in importance_map:
                importances.append(importance_map[param_name])

        if not importances:
            return 0.0
        return float(np.mean(importances))

    def _enforce_constraints(self, decisions: Dict[str, bool]) -> Dict[str, bool]:
        """Ensure min/max active ratio constraints are met."""
        total = len(decisions)
        if total == 0:
            return decisions

        active = sum(1 for v in decisions.values() if v)
        min_active = max(1, int(total * self.min_active_ratio))
        max_active = int(total * self.max_active_ratio)

        if active < min_active:
            # Activate more blocks (sort by name to be deterministic)
            inactive = sorted([n for n, v in decisions.items() if not v])
            for name in inactive[:min_active - active]:
                decisions[name] = True

        elif active > max_active:
            # Deactivate some blocks
            active_blocks = sorted([n for n, v in decisions.items() if v])
            for name in active_blocks[max_active:]:
                decisions[name] = False

        return decisions

    def apply_sparse_gradients(self, model, importance_map: Dict[str, float],
                                parameter_blocks: List[ModularParameterBlock]):
        """
        Apply gradient masking based on routing decisions.

        Zeros out gradients for frozen parameter blocks.
        """
        decisions = self.route(importance_map, parameter_blocks)

        for block in parameter_blocks:
            if not decisions.get(block.name, False):
                # Zero out gradients for frozen blocks
                block.zero_gradients()
            else:
                # Scale gradients by block importance for active blocks
                block_imp = self._compute_block_importance(block, importance_map)
                block.scale_gradients(block_imp)

    def set_threshold(self, threshold: float):
        """Update routing threshold (called by AdaptiveController)."""
        self.threshold = threshold

    def get_stats(self) -> dict:
        """Get routing statistics."""
        if not self._routing_history:
            return {'step': 0, 'avg_active_ratio': 0}

        recent = self._routing_history[-10:]
        return {
            'step': self._step,
            'avg_active_ratio': float(np.mean([r['active_ratio'] for r in recent])),
            'last_active_blocks': recent[-1]['active_blocks'] if recent else [],
            'routing_history_length': len(self._routing_history),
        }
