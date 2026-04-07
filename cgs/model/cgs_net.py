"""
CGS-Net: Cognitive Gradient Sparse Network.

The full model assembly that combines:
  - Multi-View Encoder → diverse input representations
  - Representation Space Hub → fused features
  - Task Head → classification/regression output
  - Modular Parameter Blocks → fine-grained learning control

This is the first architecture where gradients are modeled explicitly —
a gradient-aware adaptive neural system with predictive learning behavior.
"""

import numpy as np
from ..nn.module import Module
from ..nn.linear import Linear
from ..nn.activation import ReLU, GELU
from ..nn.normalization import LayerNorm
from ..nn.dropout import Dropout
from ..nn.container import Sequential, ModuleList
from ..tensor.tensor import CGSTensor
from ..encoder.multi_view import MultiViewEncoder
from ..encoder.representation_hub import RepresentationSpaceHub
from .variants import get_variant_config


class TaskHead(Module):
    """
    Task-specific output head.

    A small MLP that projects fused representations to task outputs
    (class logits, regression values, etc.).
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        self.head = Sequential(
            Linear(input_dim, hidden_dim),
            LayerNorm(hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim, num_classes),
        )

    def forward(self, x: CGSTensor) -> CGSTensor:
        return self.head(x)

    def __repr__(self):
        return f"TaskHead(\n  {self.head}\n)"


class CGSNet(Module):
    """
    CGS-Net: Cognitive Gradient Sparse Network.

    Architecture:
      Input → Multi-View Encoder → RepresentationSpaceHub → TaskHead → Output

    The model is designed to be used with the CGS training pipeline,
    which adds Gradient Probing, GID scoring, and Sparse Update Routing
    around this architecture.

    Args:
        input_dim: Input feature dimension (e.g., 784 for MNIST).
        num_classes: Number of output classes.
        variant: Model variant ('S', 'M', 'L') or custom config dict.
    """

    def __init__(self, input_dim: int, num_classes: int, variant: str = 'S'):
        super().__init__()

        # Load variant configuration
        if isinstance(variant, dict):
            self.config = variant
        else:
            self.config = get_variant_config(variant)

        self.input_dim = input_dim
        self.num_classes = num_classes
        hidden_dim = self.config['hidden_dim']
        rep_dim = self.config['rep_dim']
        dropout = self.config['dropout']

        # 1. Multi-View Encoder
        self.encoder = MultiViewEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=rep_dim,
            num_layers=self.config['encoder_layers'],
            noise_std=self.config['noise_std'],
            mask_ratio=self.config['mask_ratio'],
            dropout=dropout,
            shared_weights=self.config['shared_encoder'],
        )

        # 2. Representation Space Hub
        self.rep_hub = RepresentationSpaceHub(
            rep_dim=rep_dim,
            num_views=4,
            fusion_mode=self.config['fusion_mode'],
        )

        # 3. Task Head
        self.task_head = TaskHead(
            input_dim=rep_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Track parameter blocks for CGS sparse routing
        self._parameter_blocks = self._create_parameter_blocks()

    def _create_parameter_blocks(self) -> list:
        """
        Organize parameters into modular blocks for sparse routing.

        Returns a list of (block_name, parameter_list) tuples.
        """
        blocks = []

        # Encoder block
        encoder_params = list(self.encoder.parameters())
        if encoder_params:
            blocks.append(('encoder', encoder_params))

        # Rep hub block
        hub_params = list(self.rep_hub.parameters())
        if hub_params:
            blocks.append(('rep_hub', hub_params))

        # Task head block
        head_params = list(self.task_head.parameters())
        if head_params:
            blocks.append(('task_head', head_params))

        return blocks

    def get_parameter_blocks(self) -> list:
        """Return parameter blocks for the sparse update router."""
        return self._parameter_blocks

    def forward(self, x: CGSTensor) -> CGSTensor:
        """
        Full forward pass through CGS-Net.

        Args:
            x: Input tensor, shape (batch, input_dim).

        Returns:
            Output logits, shape (batch, num_classes).
        """
        # 1. Multi-view encoding → list of representations
        view_reps = self.encoder(x)

        # 2. Representation fusion
        fused = self.rep_hub(view_reps)

        # 3. Task-specific output
        output = self.task_head(fused)

        return output

    def forward_with_representations(self, x: CGSTensor) -> tuple:
        """
        Forward pass that also returns intermediate representations.

        Useful for gradient probing and analysis.

        Returns:
            (output, view_reps, fused_rep)
        """
        view_reps = self.encoder(x)
        fused = self.rep_hub(view_reps)
        output = self.task_head(fused)
        return output, view_reps, fused

    def __repr__(self):
        name = self.config.get('name', 'CGSNet')
        return (f"{name}(\n"
                f"  input_dim={self.input_dim}, "
                f"num_classes={self.num_classes}\n"
                f"  (encoder): {self.encoder}\n"
                f"  (rep_hub): {self.rep_hub}\n"
                f"  (task_head): {self.task_head}\n"
                f"  params={self.count_parameters():,}\n"
                f")")
