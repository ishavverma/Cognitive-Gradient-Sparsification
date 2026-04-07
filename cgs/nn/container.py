"""
Module Containers — Sequential and ModuleList.
"""

from .module import Module
from ..tensor.tensor import CGSTensor
from typing import List


class Sequential(Module):
    """
    A sequential container that chains modules.

    Modules are added in the order they are passed and executed
    sequentially in forward().

    Usage:
        model = Sequential(
            Linear(784, 256),
            ReLU(),
            Linear(256, 10),
        )
        output = model(input)
    """

    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            self.register_module(str(i), module)

    def forward(self, x: CGSTensor) -> CGSTensor:
        for module in self._modules.values():
            x = module(x)
        return x

    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx]

    def __len__(self) -> int:
        return len(self._modules)

    def __repr__(self):
        lines = ["Sequential("]
        for name, module in self._modules.items():
            lines.append(f"  ({name}): {module}")
        lines.append(")")
        return '\n'.join(lines)


class ModuleList(Module):
    """
    A list of modules that registers them for parameter tracking.

    Unlike a plain Python list, submodules in a ModuleList are
    properly registered and their parameters are discoverable.
    """

    def __init__(self, modules: List[Module] = None):
        super().__init__()
        if modules is not None:
            for i, module in enumerate(modules):
                self.register_module(str(i), module)

    def append(self, module: Module) -> 'ModuleList':
        """Add a module to the list."""
        idx = len(self._modules)
        self.register_module(str(idx), module)
        return self

    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleList does not implement forward(). "
                                  "Use individual modules.")

    def __repr__(self):
        lines = ["ModuleList("]
        for name, module in self._modules.items():
            lines.append(f"  ({name}): {module}")
        lines.append(")")
        return '\n'.join(lines)
