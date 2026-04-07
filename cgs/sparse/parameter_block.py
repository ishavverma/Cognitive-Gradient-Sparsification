"""
Modular Parameter Block (MPB).

Instead of monolithic layers, CGS-Net splits the model into independent
parameter blocks. Each block:
  - Learns specific patterns
  - Can be activated, frozen, or adapted independently
  - Has its own effective learning rate based on gradient importance

This enables fine-grained learning control and prevents overfitting.
"""

import numpy as np
from typing import List, Optional
from enum import Enum


class BlockState(Enum):
    """State of a parameter block."""
    ACTIVE = "active"       # Receiving gradient updates
    FROZEN = "frozen"       # No updates
    ADAPTING = "adapting"   # Updates scaled by importance


class ModularParameterBlock:
    """
    An independent learning unit wrapping a group of parameters.

    Args:
        name: Block identifier.
        parameters: List of CGSTensor parameters in this block.
        parameter_names: Corresponding parameter names.
    """

    def __init__(self, name: str, parameters: list, parameter_names: List[str]):
        self.name = name
        self.parameters = parameters
        self.parameter_names = parameter_names
        self.state = BlockState.ACTIVE
        self.importance = 1.0
        self._update_count = 0
        self._freeze_count = 0

    def activate(self):
        """Set block to active — allow gradient updates."""
        self.state = BlockState.ACTIVE

    def freeze(self):
        """Set block to frozen — prevent gradient updates."""
        self.state = BlockState.FROZEN
        self._freeze_count += 1

    def adapt(self, importance: float):
        """Set block to adapting — scale updates by importance."""
        self.state = BlockState.ADAPTING
        self.importance = importance

    def zero_gradients(self):
        """Zero out gradients for all parameters in this block."""
        for param in self.parameters:
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)

    def scale_gradients(self, scale: float):
        """Scale gradients by a factor."""
        for param in self.parameters:
            if param.grad is not None:
                param.grad = param.grad * scale

    def is_active(self) -> bool:
        return self.state != BlockState.FROZEN

    def get_gradient_norm(self) -> float:
        """Compute total gradient norm for this block."""
        total = 0.0
        for param in self.parameters:
            if param.grad is not None:
                total += np.sum(param.grad ** 2)
        return float(np.sqrt(total))

    def count_parameters(self) -> int:
        """Total number of parameters in this block."""
        return sum(p.data.size for p in self.parameters)

    @property
    def update_ratio(self) -> float:
        """Ratio of updates to total steps."""
        total = self._update_count + self._freeze_count
        if total == 0:
            return 1.0
        return self._update_count / total

    def step(self):
        """Record that an update happened (called after optimizer step)."""
        if self.state != BlockState.FROZEN:
            self._update_count += 1

    def __repr__(self):
        return (f"ModularParameterBlock(name='{self.name}', "
                f"state={self.state.value}, "
                f"params={self.count_parameters():,}, "
                f"update_ratio={self.update_ratio:.2f})")


def create_parameter_blocks(model, granularity: str = 'module') -> List[ModularParameterBlock]:
    """
    Create parameter blocks from a model.

    Args:
        model: A Module instance.
        granularity: 'module' (one block per submodule) or
                     'layer' (one block per parameter).

    Returns:
        List of ModularParameterBlocks.
    """
    blocks = []

    if granularity == 'module':
        # Group parameters by top-level submodule
        for mod_name, module in model._modules.items():
            params = list(module.parameters())
            param_names = [f"{mod_name}.{n}" for n, _ in module.named_parameters()]
            if params:
                blocks.append(ModularParameterBlock(mod_name, params, param_names))

    elif granularity == 'layer':
        # Each parameter is its own block
        for name, param in model.named_parameters():
            blocks.append(ModularParameterBlock(name, [param], [name]))

    elif granularity == 'auto':
        # Create blocks based on model structure with reasonable grouping
        # Group every 2-3 parameters together
        all_params = list(model.named_parameters())
        block_size = max(1, len(all_params) // 4)  # ~4 blocks

        for i in range(0, len(all_params), block_size):
            chunk = all_params[i:i + block_size]
            names = [n for n, _ in chunk]
            params = [p for _, p in chunk]
            block_name = f"block_{i // block_size}"
            blocks.append(ModularParameterBlock(block_name, params, names))

    return blocks
