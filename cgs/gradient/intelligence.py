"""
Gradient Intelligence Engine (GIE) — The Orchestrator.

Combines:
  - Gradient Probing Layer outputs
  - Gradient Information Density scoring
  - Gradient Memory Graph novelty lookups

Produces a gradient importance map that tells the Sparse Update Router
which parameters to update and which to skip.
"""

import numpy as np
from typing import Dict, Optional
from ..nn.module import Module
from ..tensor.tensor import CGSTensor
from .probing import GradientProbingLayer
from .gid import GradientInformationDensity
from .memory_graph import GradientMemoryGraph


class GradientIntelligenceEngine:
    """
    The brain of the CGS framework.

    Orchestrates gradient analysis:
      1. Probes gradients (via GPL)
      2. Scores them (via GID)
      3. Checks novelty (via GMG)
      4. Produces importance map for routing

    Args:
        alpha, beta, gamma: GID component weights.
        probe_fraction: Fraction of parameters to probe.
        memory_capacity: Max nodes per parameter in memory graph.
        use_full_probing: If True, use full probing (slower but more informative).
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.4, gamma: float = 0.3,
                 probe_fraction: float = 0.3, memory_capacity: int = 50,
                 use_full_probing: bool = True):
        self.probing_layer = GradientProbingLayer(
            probe_fraction=probe_fraction,
            probe_lr=0.01,
            num_probes=3,
        )
        self.gid_calculator = GradientInformationDensity(
            alpha=alpha, beta=beta, gamma=gamma,
        )
        self.memory_graph = GradientMemoryGraph(
            max_nodes_per_param=memory_capacity,
        )
        self.use_full_probing = use_full_probing

        # Statistics tracking
        self._step = 0
        self._history = {
            'avg_gid': [],
            'sparsity': [],
            'avg_magnitude': [],
            'avg_novelty': [],
        }

    def analyze(self, model: Module, loss_fn, x: CGSTensor,
                targets: CGSTensor) -> Dict[str, dict]:
        """
        Full gradient intelligence analysis.

        Performs probing → GID scoring → novelty lookup → importance mapping.

        Args:
            model: The neural network model.
            loss_fn: Loss function.
            x: Input batch.
            targets: Target labels.

        Returns:
            intelligence_report: Dict mapping parameter names to:
              - 'gid_score': float — overall importance
              - 'gradient': ndarray — the computed gradient
              - 'magnitude': float
              - 'novelty': float
              - 'impact': float
              - 'should_update': bool — recommendation
        """
        self._step += 1
        report = {}

        if self.use_full_probing:
            # Full probing with simulation
            preview = self.probing_layer.probe(model, loss_fn, x, targets)
            gid_scores = self.gid_calculator.compute_gid(preview, self.memory_graph)

            for name, info in preview.items():
                grad = info['gradient']
                gid = gid_scores.get(name, 0.0)
                novelty = self.memory_graph.query_novelty(name, grad)

                report[name] = {
                    'gid_score': gid,
                    'gradient': grad,
                    'magnitude': info['magnitude'],
                    'novelty': novelty,
                    'impact': abs(info.get('predicted_loss_change', 0.0)),
                    'stability': info.get('direction_stability', 1.0),
                    'should_update': True,  # Will be decided by router
                }

                # Update memory graph
                self.memory_graph.add(name, grad, {
                    'step': self._step,
                    'gid': gid,
                    'loss_change': info.get('predicted_loss_change', 0.0),
                })
        else:
            # Quick analysis without full probing
            gradients = self.probing_layer.quick_probe(model, loss_fn, x, targets)
            gid_scores = self.gid_calculator.compute_gid_simple(gradients, self.memory_graph)

            for name, grad in gradients.items():
                gid = gid_scores.get(name, 0.0)
                magnitude = float(np.sqrt(np.sum(grad ** 2)))
                novelty = self.memory_graph.query_novelty(name, grad)

                report[name] = {
                    'gid_score': gid,
                    'gradient': grad,
                    'magnitude': magnitude,
                    'novelty': novelty,
                    'impact': magnitude,  # Use magnitude as proxy
                    'stability': 1.0,
                    'should_update': True,
                }

                self.memory_graph.add(name, grad, {
                    'step': self._step,
                    'gid': gid,
                })

        # Track statistics
        if report:
            gid_values = [r['gid_score'] for r in report.values()]
            self._history['avg_gid'].append(float(np.mean(gid_values)))
            self._history['avg_magnitude'].append(
                float(np.mean([r['magnitude'] for r in report.values()]))
            )
            self._history['avg_novelty'].append(
                float(np.mean([r['novelty'] for r in report.values()]))
            )

        return report

    def get_importance_map(self, report: Dict[str, dict]) -> Dict[str, float]:
        """Extract just the GID scores from a full report."""
        return {name: info['gid_score'] for name, info in report.items()}

    def get_stats(self) -> dict:
        """Return analysis statistics."""
        return {
            'step': self._step,
            'memory_stats': self.memory_graph.get_stats(),
            'history': {k: v[-10:] for k, v in self._history.items()},
            'gid_weights': self.gid_calculator.get_weights(),
        }

    def update_gid_weights(self, alpha: float, beta: float, gamma: float):
        """Update GID component weights (called by AdaptiveController)."""
        self.gid_calculator.set_weights(alpha, beta, gamma)

    def reset(self):
        """Reset all internal state."""
        self.memory_graph.clear()
        self.gid_calculator.reset_stats()
        self._step = 0
        self._history = {k: [] for k in self._history}
