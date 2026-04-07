"""
Multi-View Encoder — Context Diversity Engine.

Creates multiple views of the same input (raw, augmented, masked, noisy)
and passes each through shared-weight encoder branches. This forces the
model to learn robust representations and creates gradient diversity,
which is critical for CGS gradient intelligence.
"""

import numpy as np
from ..nn.module import Module
from ..nn.linear import Linear
from ..nn.activation import ReLU, GELU
from ..nn.normalization import LayerNorm
from ..nn.dropout import Dropout
from ..nn.container import Sequential, ModuleList
from ..tensor.tensor import CGSTensor
from ..data.transforms import AddGaussianNoise, RandomMask, RandomScale


class EncoderBranch(Module):
    """
    A single encoder branch — a small MLP with normalization.

    All branches share the same architecture but NOT weights by default.
    Set shared_weights=True to use the same parameters across views.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(Linear(in_dim, hidden_dim))
            layers.append(LayerNorm(hidden_dim))
            layers.append(GELU())
            layers.append(Dropout(dropout))
            in_dim = hidden_dim

        layers.append(Linear(in_dim, output_dim))
        layers.append(LayerNorm(output_dim))

        self.network = Sequential(*layers)

    def forward(self, x: CGSTensor) -> CGSTensor:
        return self.network(x)

    def __repr__(self):
        return f"EncoderBranch(\n  {self.network}\n)"


class MultiViewEncoder(Module):
    """
    Multi-View Encoder (Context Diversity Engine).

    Takes a single input and creates 4 views:
      1. Raw — original input
      2. Augmented — noisy version
      3. Masked — randomly masked version
      4. Scaled — randomly scaled version

    Each view is passed through a shared-weight encoder branch,
    producing a list of representation tensors.

    Args:
        input_dim: Dimension of input features.
        hidden_dim: Hidden layer dimension in encoder branches.
        output_dim: Output representation dimension.
        num_layers: Number of layers per branch.
        noise_std: Standard deviation for noise augmentation.
        mask_ratio: Fraction of features to mask.
        dropout: Dropout rate in encoder.
        shared_weights: If True, all branches share the same weights.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, noise_std: float = 0.1,
                 mask_ratio: float = 0.15, dropout: float = 0.1,
                 shared_weights: bool = True):
        super().__init__()
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio
        self.shared_weights = shared_weights
        self.num_views = 4

        if shared_weights:
            # Shared encoder for all views
            self.shared_encoder = EncoderBranch(input_dim, hidden_dim, output_dim,
                                                 num_layers, dropout)
        else:
            # Independent encoder per view
            self.encoders = ModuleList([
                EncoderBranch(input_dim, hidden_dim, output_dim, num_layers, dropout)
                for _ in range(self.num_views)
            ])

        # View-specific transforms (applied to numpy data before tensor conversion)
        self._noise_transform = AddGaussianNoise(noise_std)
        self._mask_transform = RandomMask(mask_ratio)

    def _create_views(self, x: CGSTensor) -> list:
        """
        Create multiple views of the input data.

        Returns list of CGSTensor views:
          [raw, noisy, masked, scaled]
        """
        raw = x

        if self._training:
            # Create augmented views by modifying the underlying data
            noisy_data = self._noise_transform(x.data)
            noisy = CGSTensor(noisy_data, requires_grad=x.requires_grad)

            masked_data = self._mask_transform(x.data)
            masked = CGSTensor(masked_data, requires_grad=x.requires_grad)

            scale = np.random.uniform(0.8, 1.2)
            scaled_data = x.data * scale
            scaled = CGSTensor(scaled_data, requires_grad=x.requires_grad)
        else:
            # During eval, only use raw view (no stochastic augmentation)
            noisy = x
            masked = x
            scaled = x

        return [raw, noisy, masked, scaled]

    def forward(self, x: CGSTensor) -> list:
        """
        Forward pass: create views → encode each.

        Args:
            x: Input tensor, shape (batch, input_dim).

        Returns:
            List of 4 representation tensors, each shape (batch, output_dim).
        """
        views = self._create_views(x)
        representations = []

        for i, view in enumerate(views):
            if self.shared_weights:
                rep = self.shared_encoder(view)
            else:
                rep = self.encoders[i](view)
            representations.append(rep)

        return representations

    def __repr__(self):
        if self.shared_weights:
            return f"MultiViewEncoder(shared=True, views={self.num_views})"
        return f"MultiViewEncoder(shared=False, views={self.num_views})"
