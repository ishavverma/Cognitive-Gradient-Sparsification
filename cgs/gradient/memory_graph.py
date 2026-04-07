"""
Gradient Memory Graph (GMG) — Graph-structured gradient history.

Not just a list — a graph where:
  - Nodes represent gradient snapshots
  - Edges represent similarity/dependency between gradients

This enables:
  - Pattern detection in learning
  - Avoiding redundant learning cycles
  - Memory-aware novelty computation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import heapq


class GradientNode:
    """A node in the Gradient Memory Graph."""

    def __init__(self, name: str, gradient: np.ndarray, step: int,
                 metadata: Optional[dict] = None):
        self.name = name  # Parameter name
        self.gradient_compressed = self._compress(gradient)  # Compressed gradient
        self.magnitude = float(np.sqrt(np.sum(gradient ** 2)))
        self.step = step  # Training step when recorded
        self.metadata = metadata or {}
        self.access_count = 0
        self.last_accessed = step

    @staticmethod
    def _compress(gradient: np.ndarray, max_dim: int = 128) -> np.ndarray:
        """
        Compress gradient for memory-efficient storage.

        Uses random projection when gradient is large.
        """
        flat = gradient.flatten()
        if len(flat) <= max_dim:
            return flat.copy()

        # Random projection for dimensionality reduction
        np.random.seed(42)  # Fixed seed for reproducibility
        proj_matrix = np.random.randn(max_dim, len(flat)) / np.sqrt(max_dim)
        compressed = proj_matrix @ flat
        return compressed

    def similarity(self, other_gradient: np.ndarray) -> float:
        """Cosine similarity between this node and another gradient."""
        other_compressed = self._compress(other_gradient)

        # Ensure same length
        min_len = min(len(self.gradient_compressed), len(other_compressed))
        a = self.gradient_compressed[:min_len]
        b = other_compressed[:min_len]

        norm_a = np.sqrt(np.sum(a ** 2)) + 1e-8
        norm_b = np.sqrt(np.sum(b ** 2)) + 1e-8

        return float(np.sum(a * b) / (norm_a * norm_b))


class GradientMemoryGraph:
    """
    Graph-structured gradient memory for novelty detection and pattern recognition.

    Maintains a bounded-size memory of past gradients per parameter,
    organized as a graph with similarity edges.

    Args:
        max_nodes_per_param: Maximum nodes to store per parameter.
        similarity_threshold: Minimum cosine similarity for edge creation.
        max_total_nodes: Global node limit.
    """

    def __init__(self, max_nodes_per_param: int = 50,
                 similarity_threshold: float = 0.7,
                 max_total_nodes: int = 500):
        self.max_nodes_per_param = max_nodes_per_param
        self.similarity_threshold = similarity_threshold
        self.max_total_nodes = max_total_nodes

        # Nodes organized by parameter name
        self._nodes: Dict[str, List[GradientNode]] = defaultdict(list)
        # Edges: (param, idx_a) -> [(param, idx_b, similarity)]
        self._edges: Dict[Tuple[str, int], List[Tuple[str, int, float]]] = defaultdict(list)
        self._total_nodes = 0
        self._step = 0

    def add(self, name: str, gradient: np.ndarray, metadata: Optional[dict] = None):
        """
        Add a gradient snapshot to the memory graph.

        Args:
            name: Parameter name.
            gradient: Gradient array.
            metadata: Optional metadata (loss, epoch, etc.).
        """
        self._step += 1
        node = GradientNode(name, gradient, self._step, metadata)

        # Compute edges to existing nodes for this parameter
        nodes = self._nodes[name]
        new_idx = len(nodes)

        for i, existing_node in enumerate(nodes):
            sim = node.similarity(existing_node.gradient_compressed)
            if abs(sim) >= self.similarity_threshold:
                self._edges[(name, new_idx)].append((name, i, sim))
                self._edges[(name, i)].append((name, new_idx, sim))

        nodes.append(node)
        self._total_nodes += 1

        # Prune if needed
        if len(nodes) > self.max_nodes_per_param:
            self._prune_param(name)

        if self._total_nodes > self.max_total_nodes:
            self._prune_global()

    def query_novelty(self, name: str, gradient: np.ndarray) -> float:
        """
        Compute novelty score for a gradient.

        Novelty = 1 - max_similarity_to_existing_nodes.
        High novelty = gradient is very different from anything seen before.

        Returns:
            Novelty score in [0, 1].
        """
        nodes = self._nodes.get(name, [])
        if not nodes:
            return 1.0  # First gradient is maximally novel

        # Find max similarity to any existing node
        max_sim = 0.0
        for node in nodes:
            sim = node.similarity(gradient)
            max_sim = max(max_sim, abs(sim))
            node.access_count += 1
            node.last_accessed = self._step

        return 1.0 - max_sim

    def get_patterns(self, name: str) -> dict:
        """
        Detect patterns in gradient history for a parameter.

        Returns:
            Pattern analysis including cluster count, repetition rate,
            and trend direction.
        """
        nodes = self._nodes.get(name, [])
        if len(nodes) < 3:
            return {'clusters': 0, 'repetition_rate': 0.0, 'trend': 'insufficient_data'}

        # Simple cluster detection via similarity
        similarities = []
        for i in range(len(nodes) - 1):
            sim = nodes[i].similarity(nodes[i+1].gradient_compressed)
            similarities.append(sim)

        avg_sim = np.mean(similarities)
        repetition_rate = float(np.mean([1 for s in similarities if s > 0.9]))

        # Magnitude trend
        magnitudes = [n.magnitude for n in nodes[-10:]]  # Last 10
        if len(magnitudes) >= 2:
            trend_slope = (magnitudes[-1] - magnitudes[0]) / len(magnitudes)
            if trend_slope > 0.01:
                trend = 'increasing'
            elif trend_slope < -0.01:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'unknown'

        # Estimate cluster count (consecutive similar gradients)
        clusters = 1
        for i, sim in enumerate(similarities):
            if sim < 0.5:
                clusters += 1

        return {
            'clusters': clusters,
            'repetition_rate': repetition_rate,
            'trend': trend,
            'avg_similarity': float(avg_sim),
            'num_nodes': len(nodes),
        }

    def _prune_param(self, name: str):
        """
        Prune nodes for a parameter using LRU + low-connectivity heuristic.
        """
        nodes = self._nodes[name]
        if len(nodes) <= self.max_nodes_per_param:
            return

        # Score each node: lower = more prunable
        scores = []
        for i, node in enumerate(nodes):
            recency = node.last_accessed / (self._step + 1)
            access_freq = node.access_count / (self._step + 1)
            connectivity = len(self._edges.get((name, i), []))
            score = recency * 0.4 + access_freq * 0.3 + connectivity * 0.3 / max(1, len(nodes))
            scores.append((score, i))

        # Keep top nodes
        scores.sort(reverse=True)
        keep_indices = set(idx for _, idx in scores[:self.max_nodes_per_param])

        new_nodes = [n for i, n in enumerate(nodes) if i in keep_indices]
        removed = self._total_nodes - len(new_nodes) - sum(
            len(v) for k, v in self._nodes.items() if k != name
        )

        self._nodes[name] = new_nodes
        self._total_nodes = sum(len(v) for v in self._nodes.values())

        # Clean up edges (simplified — rebuild for this param)
        keys_to_remove = [k for k in self._edges if k[0] == name]
        for k in keys_to_remove:
            del self._edges[k]

    def _prune_global(self):
        """Global pruning across all parameters."""
        while self._total_nodes > self.max_total_nodes:
            # Find parameter with most nodes
            max_param = max(self._nodes.keys(), key=lambda k: len(self._nodes[k]))
            # Remove oldest node
            if self._nodes[max_param]:
                self._nodes[max_param].pop(0)
                self._total_nodes -= 1

    def get_stats(self) -> dict:
        """Get memory graph statistics."""
        return {
            'total_nodes': self._total_nodes,
            'num_parameters_tracked': len(self._nodes),
            'total_edges': sum(len(v) for v in self._edges.values()) // 2,
            'nodes_per_param': {k: len(v) for k, v in self._nodes.items()},
            'step': self._step,
        }

    def clear(self):
        """Clear all memory."""
        self._nodes.clear()
        self._edges.clear()
        self._total_nodes = 0
