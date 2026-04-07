"""
Representation Space Hub (RSH).

A central latent interaction layer where all encoder outputs interact.
Representations are fused using similarity, contrast, and alignment —
think of it as a "negotiation space" between features from different views.
"""

import numpy as np
from ..nn.module import Module
from ..nn.linear import Linear
from ..nn.activation import GELU
from ..nn.normalization import LayerNorm
from ..tensor.tensor import CGSTensor
from ..tensor.ops import matmul, add, relu, tensor_mean, tensor_sum, reshape
from ..tensor import functional as F


class RepresentationSpaceHub(Module):
    """
    Representation Space Hub — Fusion via Similarity, Contrast, Alignment.

    Receives multiple view representations and produces a unified
    representation by computing pairwise interactions and fusing them.

    Three fusion mechanisms:
      1. Similarity: Cosine similarity-weighted averaging
      2. Contrast: Difference-based features between views
      3. Alignment: Learned projection into shared space

    Args:
        rep_dim: Dimension of each representation vector.
        num_views: Number of views to fuse.
        fusion_mode: Fusion strategy ('attention', 'weighted', 'concat').
    """

    def __init__(self, rep_dim: int, num_views: int = 4,
                 fusion_mode: str = 'attention'):
        super().__init__()
        self.rep_dim = rep_dim
        self.num_views = num_views
        self.fusion_mode = fusion_mode

        if fusion_mode == 'attention':
            # Attention-based fusion: learn query to compute view importance
            self.query = CGSTensor(
                np.random.randn(1, rep_dim) * 0.02,
                requires_grad=True, name='attention_query'
            )
            self.attn_proj = Linear(rep_dim, rep_dim)
        elif fusion_mode == 'concat':
            # Concatenate all views and project
            self.fusion_proj = Linear(rep_dim * num_views, rep_dim)
        elif fusion_mode == 'weighted':
            # Learnable view weights
            self.view_weights = CGSTensor(
                np.ones(num_views) / num_views,
                requires_grad=True, name='view_weights'
            )

        # Shared post-fusion layers
        self.norm = LayerNorm(rep_dim)
        self.output_proj = Linear(rep_dim, rep_dim)
        self.activation = GELU()

    def _compute_similarity_matrix(self, representations: list) -> np.ndarray:
        """
        Compute pairwise cosine similarity between view representations.

        Returns (num_views, num_views) similarity matrix (averaged over batch).
        """
        sims = np.zeros((self.num_views, self.num_views))
        for i in range(self.num_views):
            for j in range(i, self.num_views):
                cos = F.cosine_similarity(representations[i], representations[j])
                sim_val = np.mean(cos.data)
                sims[i, j] = sim_val
                sims[j, i] = sim_val
        return sims

    def forward(self, representations: list) -> CGSTensor:
        """
        Fuse multiple view representations into a unified representation.

        Args:
            representations: List of CGSTensors, each shape (batch, rep_dim).

        Returns:
            Fused representation, shape (batch, rep_dim).
        """
        if self.fusion_mode == 'attention':
            return self._attention_fusion(representations)
        elif self.fusion_mode == 'concat':
            return self._concat_fusion(representations)
        elif self.fusion_mode == 'weighted':
            return self._weighted_fusion(representations)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

    def _attention_fusion(self, reps: list) -> CGSTensor:
        """
        Attention-based fusion: use learned query to weight views.
        """
        batch_size = reps[0].shape[0]

        # Stack representations: (batch, num_views, rep_dim)
        stacked_data = np.stack([r.data for r in reps], axis=1)
        stacked = CGSTensor(stacked_data, requires_grad=True)

        # Compute attention scores using learned query
        # scores = stacked @ query^T → (batch, num_views, 1)
        projected = self.attn_proj(CGSTensor(stacked_data.reshape(-1, self.rep_dim)))
        projected_data = projected.data.reshape(batch_size, self.num_views, self.rep_dim)

        # Attention scores
        scores = np.sum(projected_data * self.query.data, axis=-1)  # (batch, num_views)
        scores = scores / np.sqrt(self.rep_dim)

        # Softmax attention weights
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)  # (batch, num_views)

        # Weighted sum of representations
        fused_data = np.sum(attn_weights[:, :, None] * stacked_data, axis=1)  # (batch, rep_dim)
        fused = CGSTensor(fused_data, requires_grad=True)

        # Manually set up backward through the attention mechanism
        track = any(r.requires_grad for r in reps) or self.query.requires_grad
        out_data = fused_data.copy()
        result = CGSTensor(out_data, requires_grad=track,
                          _prev=set(reps) | {self.query, projected})

        if track:
            def _backward(grad):
                # Gradient flows to each representation proportional to attention weight
                for v in range(self.num_views):
                    if reps[v].requires_grad:
                        r_grad = grad * attn_weights[:, v:v+1]
                        reps[v].grad = r_grad if reps[v].grad is None else reps[v].grad + r_grad
            result._backward_fn = _backward

        # Post-fusion processing
        result = self.norm(result)
        result = self.output_proj(result)
        result = self.activation(result)

        return result

    def _concat_fusion(self, reps: list) -> CGSTensor:
        """Concatenate all views and project."""
        from ..tensor.ops import concatenate
        concat = concatenate(reps, axis=-1)
        fused = self.fusion_proj(concat)
        fused = self.norm(fused)
        fused = self.activation(fused)
        return fused

    def _weighted_fusion(self, reps: list) -> CGSTensor:
        """Weighted sum fusion with learnable weights."""
        # Softmax the view weights
        w = np.exp(self.view_weights.data)
        w = w / np.sum(w)

        # Weighted sum
        fused_data = np.zeros_like(reps[0].data)
        for i, rep in enumerate(reps):
            fused_data += w[i] * rep.data

        fused = CGSTensor(fused_data, requires_grad=True, _prev=set(reps) | {self.view_weights})

        track = any(r.requires_grad for r in reps)
        if track:
            def _backward(grad):
                for i, rep in enumerate(reps):
                    if rep.requires_grad:
                        r_grad = grad * w[i]
                        rep.grad = r_grad if rep.grad is None else rep.grad + r_grad
            fused._backward_fn = _backward

        fused = self.norm(fused)
        fused = self.output_proj(fused)
        fused = self.activation(fused)
        return fused

    def __repr__(self):
        return (f"RepresentationSpaceHub(dim={self.rep_dim}, "
                f"views={self.num_views}, mode='{self.fusion_mode}')")
