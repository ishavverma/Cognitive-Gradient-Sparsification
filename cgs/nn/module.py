"""
Base Module class for all neural network components.

Provides parameter management, state serialization, and the
forward/backward interface that all layers implement.
"""

import numpy as np
from collections import OrderedDict
from typing import Iterator, Tuple, Dict, Any
from ..tensor.tensor import CGSTensor


class Module:
    """
    Base class for all neural network modules.

    Every layer, model, and trainable component inherits from Module.
    Provides:
      - Automatic parameter discovery via named_parameters()
      - State dict save/load for serialization
      - Training/eval mode switching
      - Recursive zero_grad
    """

    def __init__(self):
        self._parameters: OrderedDict[str, CGSTensor] = OrderedDict()
        self._modules: OrderedDict[str, 'Module'] = OrderedDict()
        self._training: bool = True

    def forward(self, *args, **kwargs):
        """Forward pass — must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, *args, **kwargs):
        """Make module callable — invokes forward()."""
        return self.forward(*args, **kwargs)

    # ──────────────────── Parameter Management ────────────────────

    def register_parameter(self, name: str, param: CGSTensor):
        """Register a parameter with this module."""
        self._parameters[name] = param

    def register_module(self, name: str, module: 'Module'):
        """Register a submodule."""
        self._modules[name] = module

    def __setattr__(self, name: str, value):
        if isinstance(value, CGSTensor) and value.requires_grad:
            if '_parameters' not in self.__dict__:
                super().__setattr__('_parameters', OrderedDict())
            self._parameters[name] = value
        elif isinstance(value, Module):
            if '_modules' not in self.__dict__:
                super().__setattr__('_modules', OrderedDict())
            self._modules[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__ and name in self.__dict__['_parameters']:
            return self.__dict__['_parameters'][name]
        if '_modules' in self.__dict__ and name in self.__dict__['_modules']:
            return self.__dict__['_modules'][name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def parameters(self) -> Iterator[CGSTensor]:
        """Yield all parameters (own + children), recursively."""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix: str = '') -> Iterator[Tuple[str, CGSTensor]]:
        """Yield (name, parameter) pairs recursively."""
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param
        for mod_name, module in self._modules.items():
            mod_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
            yield from module.named_parameters(mod_prefix)

    def named_modules(self, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
        """Yield (name, module) pairs recursively including self."""
        yield prefix, self
        for name, module in self._modules.items():
            mod_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(mod_prefix)

    # ──────────────────── Gradient ────────────────────

    def zero_grad(self):
        """Zero all parameter gradients recursively."""
        for param in self.parameters():
            param.zero_grad()

    # ──────────────────── Training / Eval ────────────────────

    @property
    def training(self) -> bool:
        return self._training

    def train(self, mode: bool = True) -> 'Module':
        """Set training mode (affects dropout, batchnorm, etc.)."""
        self._training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> 'Module':
        """Set evaluation mode."""
        return self.train(False)

    # ──────────────────── Serialization ────────────────────

    def state_dict(self) -> Dict[str, np.ndarray]:
        """Return a dictionary of all parameter data (for saving)."""
        state = OrderedDict()
        for name, param in self.named_parameters():
            state[name] = param.data.copy()
        return state

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]):
        """Load parameters from a state dictionary."""
        param_dict = dict(self.named_parameters())
        for name, data in state_dict.items():
            if name in param_dict:
                param_dict[name].data = data.copy()
            else:
                raise KeyError(f"Unexpected key '{name}' in state_dict")

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.data.size for p in self.parameters())

    # ──────────────────── Representation ────────────────────

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for name, module in self._modules.items():
            mod_repr = repr(module).replace('\n', '\n  ')
            lines.append(f"  ({name}): {mod_repr}")
        for name, param in self._parameters.items():
            lines.append(f"  ({name}): Parameter{param.shape}")
        lines.append(")")
        return '\n'.join(lines)
