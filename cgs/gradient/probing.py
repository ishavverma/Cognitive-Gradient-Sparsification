"""
Gradient Probing Layer (GPL) — "Simulate Before You Learn".

The core novelty of CGS-Net. Before full backpropagation, the model
performs partial gradient simulations to preview what would happen
if it updated specific parameters. This enables informed decision-making
about which gradients are worth applying.

Think of it as a "dry-run" of learning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..tensor.tensor import CGSTensor, no_grad
from ..nn.module import Module


class GradientProbingLayer:
    """
    Gradient Probing Layer — simulates gradient updates before committing.

    Instead of computing gradients and blindly applying them, GPL:
      1. Performs a forward pass
      2. Computes gradients for each parameter block
      3. Simulates what the loss would be after applying those gradients
      4. Returns a gradient preview map showing the predicted impact

    This is NOT a neural network layer — it's a meta-learning component
    that wraps around the training loop.

    Args:
        probe_fraction: Fraction of parameters to probe (for efficiency).
        probe_lr: Learning rate used in simulated updates.
        num_probes: Number of probe variations to try.
    """

    def __init__(self, probe_fraction: float = 0.3, probe_lr: float = 0.01,
                 num_probes: int = 3):
        self.probe_fraction = probe_fraction
        self.probe_lr = probe_lr
        self.num_probes = num_probes

    def probe(self, model: Module, loss_fn, x: CGSTensor, targets: CGSTensor) -> Dict[str, dict]:
        """
        Perform gradient probing on the model.

        For each parameter block, simulates a gradient update and measures
        the predicted change in loss.

        Args:
            model: The CGS-Net model.
            loss_fn: Loss function (takes logits, targets → scalar loss).
            x: Input batch.
            targets: Target labels.

        Returns:
            gradient_preview_map: Dict mapping parameter names to:
              - 'gradient': The computed gradient (ndarray)
              - 'magnitude': L2 norm of gradient
              - 'predicted_loss_change': Estimated Δloss if this gradient is applied
              - 'direction_stability': How consistent the gradient direction is
        """
        preview_map = {}

        # Step 1: Compute current loss (no_grad for efficiency)
        with no_grad():
            current_output = model(x)
        current_loss_val = loss_fn(model(x), targets).data.item()

        # Step 2: Full gradient computation
        model.zero_grad()
        output = model(x)
        loss = loss_fn(output, targets)
        loss.backward()

        # Step 3: For each parameter, compute gradient statistics and simulate update
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.copy()
            magnitude = float(np.sqrt(np.sum(grad ** 2)))

            # Skip very small gradients
            if magnitude < 1e-10:
                preview_map[name] = {
                    'gradient': grad,
                    'magnitude': magnitude,
                    'predicted_loss_change': 0.0,
                    'direction_stability': 0.0,
                }
                continue

            # Simulate update: temporarily apply gradient and measure new loss
            predicted_loss_change = self._simulate_update(
                model, param, grad, loss_fn, x, targets, current_loss_val
            )

            # Probe direction stability via multiple small perturbations
            direction_stability = self._probe_stability(
                model, param, grad, loss_fn, x, targets
            )

            preview_map[name] = {
                'gradient': grad,
                'magnitude': magnitude,
                'predicted_loss_change': predicted_loss_change,
                'direction_stability': direction_stability,
            }

        return preview_map

    def _simulate_update(self, model: Module, param: CGSTensor,
                         grad: np.ndarray, loss_fn, x: CGSTensor,
                         targets: CGSTensor, current_loss: float) -> float:
        """
        Simulate applying a gradient update and measure loss change.

        Temporarily modifies the parameter, computes loss, then restores.
        """
        # Save original
        original_data = param.data.copy()

        # Apply simulated update
        param.data = original_data - self.probe_lr * grad

        # Measure new loss
        with no_grad():
            new_output = model(x)
            new_loss = loss_fn(new_output, targets).data.item()

        # Restore original
        param.data = original_data

        return current_loss - new_loss  # Positive = loss would decrease

    def _probe_stability(self, model: Module, param: CGSTensor,
                         grad: np.ndarray, loss_fn, x: CGSTensor,
                         targets: CGSTensor) -> float:
        """
        Probe gradient direction stability by testing multiple perturbations.

        High stability = the gradient consistently points in the same
        beneficial direction. Low stability = the gradient is noisy.
        """
        if self.num_probes <= 1:
            return 1.0

        original_data = param.data.copy()
        loss_changes = []

        for i in range(self.num_probes):
            # Apply gradient with different step sizes
            scale = (i + 1) / self.num_probes
            param.data = original_data - self.probe_lr * scale * grad

            with no_grad():
                out = model(x)
                new_loss = loss_fn(out, targets).data.item()

            loss_changes.append(new_loss)
            param.data = original_data

        # If losses consistently decrease, gradient is stable
        # Compute monotonicity score
        if len(loss_changes) < 2:
            return 1.0

        decreasing = sum(1 for i in range(1, len(loss_changes))
                        if loss_changes[i] <= loss_changes[i-1])
        stability = decreasing / (len(loss_changes) - 1)

        return stability

    def quick_probe(self, model: Module, loss_fn, x: CGSTensor,
                    targets: CGSTensor) -> Dict[str, np.ndarray]:
        """
        Quick probe: just compute gradients without simulation.

        Faster but less informative than full probe.
        """
        model.zero_grad()
        output = model(x)
        loss = loss_fn(output, targets)
        loss.backward()

        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.copy()

        return gradients
