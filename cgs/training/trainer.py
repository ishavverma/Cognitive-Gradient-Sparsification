"""
CGSTrainer — The End-to-End CGS Training Pipeline.

Implements the full closed feedback loop:
  1. Multi-view encode input
  2. Forward through RepresentationSpaceHub
  3. Compute task loss
  4. Gradient Probing Layer simulates updates
  5. GIE scores all gradients (GID)
  6. Sparsifier filters gradients
  7. SUR routes updates to parameter blocks
  8. Optimizer steps on selected gradients only
  9. Memory graph updated
  10. Adaptive controller adjusts hyperparams
"""

import numpy as np
import time
from typing import Optional, List, Dict
from ..tensor.tensor import CGSTensor
from ..nn.module import Module
from ..nn.loss import CrossEntropyLoss
from ..optim.optimizer import Optimizer
from ..data.dataloader import DataLoader
from ..gradient.intelligence import GradientIntelligenceEngine
from ..sparse.sparsifier import GradientSparsifier
from ..sparse.router import SparseUpdateRouter
from ..sparse.parameter_block import ModularParameterBlock, create_parameter_blocks
from ..controller.adaptive import AdaptiveController


class CGSTrainer:
    """
    Complete CGS training pipeline.

    Orchestrates all CGS components into a unified training loop that
    implements gradient-aware selective learning.

    Args:
        model: CGSNet model.
        optimizer: Optimizer (SGD, Adam).
        loss_fn: Loss function module.
        use_cgs: If True, use CGS gradient intelligence. If False, standard training.
        use_full_probing: If True, use full gradient probing (slower but better).
        gid_alpha, gid_beta, gid_gamma: GID component weights.
        sparsity_mode: Sparsification mode ('threshold', 'topk', 'hybrid').
        sparsity_threshold: Initial sparsity threshold.
        warmup_epochs: Number of warm-up epochs.
        callbacks: List of callback instances.
        block_granularity: Parameter block granularity ('module', 'layer', 'auto').
    """

    def __init__(self, model: Module, optimizer: Optimizer, loss_fn=None,
                 use_cgs: bool = True, use_full_probing: bool = False,
                 gid_alpha: float = 0.3, gid_beta: float = 0.4, gid_gamma: float = 0.3,
                 sparsity_mode: str = 'hybrid', sparsity_threshold: float = 0.3,
                 warmup_epochs: int = 2, callbacks: Optional[list] = None,
                 block_granularity: str = 'module'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn or CrossEntropyLoss()
        self.use_cgs = use_cgs
        self.callbacks = callbacks or []

        # CGS Components
        if use_cgs:
            self.intelligence = GradientIntelligenceEngine(
                alpha=gid_alpha, beta=gid_beta, gamma=gid_gamma,
                use_full_probing=use_full_probing,
            )
            self.sparsifier = GradientSparsifier(
                mode=sparsity_mode,
                threshold=sparsity_threshold,
            )
            self.router = SparseUpdateRouter(
                threshold=sparsity_threshold,
                routing_mode='adaptive',
            )
            self.controller = AdaptiveController(
                initial_threshold=sparsity_threshold,
                initial_lr=optimizer.lr,
                warmup_epochs=warmup_epochs,
            )
            self.parameter_blocks = create_parameter_blocks(model, block_granularity)
        else:
            self.intelligence = None
            self.sparsifier = None
            self.router = None
            self.controller = None
            self.parameter_blocks = []

        # Training state
        self._epoch = 0
        self._global_step = 0
        self._history = {
            'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
            'sparsity': [], 'lr': [], 'avg_gid': [], 'epoch_time': [],
        }

    def train(self, train_loader: DataLoader, epochs: int,
              val_loader: Optional[DataLoader] = None,
              log_interval: int = 50) -> dict:
        """
        Full training loop.

        Args:
            train_loader: Training data loader.
            epochs: Number of training epochs.
            val_loader: Optional validation data loader.
            log_interval: Log every N batches.

        Returns:
            Training history dictionary.
        """
        print(f"\n{'='*60}")
        print(f"  CGS Training {'(ENABLED)' if self.use_cgs else '(DISABLED — Standard)'}")
        print(f"  Model: {self.model.__class__.__name__}")
        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Epochs: {epochs}")
        print(f"{'='*60}\n")

        for cb in self.callbacks:
            cb.on_train_begin(self)

        for epoch in range(epochs):
            self._epoch = epoch
            epoch_start = time.time()

            # Training phase
            self.model.train()
            train_metrics = self._train_epoch(train_loader, epoch, log_interval)

            # Validation phase
            val_metrics = {}
            if val_loader is not None:
                self.model.eval()
                val_metrics = self._validate(val_loader)

            epoch_time = time.time() - epoch_start

            # Record history
            self._history['train_loss'].append(train_metrics['loss'])
            self._history['train_acc'].append(train_metrics['accuracy'])
            self._history['epoch_time'].append(epoch_time)
            if val_metrics:
                self._history['val_loss'].append(val_metrics['loss'])
                self._history['val_acc'].append(val_metrics['accuracy'])
            if self.use_cgs:
                self._history['sparsity'].append(train_metrics.get('avg_sparsity', 0))
                self._history['avg_gid'].append(train_metrics.get('avg_gid', 0))
            self._history['lr'].append(self.optimizer.lr)

            # Adaptive controller epoch step
            if self.controller:
                self.controller.epoch_step(epoch, val_metrics.get('loss'))

            # Print epoch summary
            self._print_epoch_summary(epoch, epochs, train_metrics, val_metrics, epoch_time)

            # Callbacks
            for cb in self.callbacks:
                stop = cb.on_epoch_end(self, epoch, train_metrics, val_metrics)
                if stop:
                    print(f"\n  ⛔ Early stopping triggered at epoch {epoch + 1}")
                    return self._history

        for cb in self.callbacks:
            cb.on_train_end(self)

        print(f"\n{'='*60}")
        print(f"  Training Complete!")
        print(f"  Final Train Acc: {self._history['train_acc'][-1]:.2%}")
        if self._history['val_acc']:
            print(f"  Final Val Acc:   {self._history['val_acc'][-1]:.2%}")
        print(f"{'='*60}\n")

        return self._history

    def _train_epoch(self, loader: DataLoader, epoch: int, log_interval: int) -> dict:
        """Train for one epoch."""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_sparsity = 0.0
        total_gid = 0.0
        num_batches = 0

        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            self._global_step += 1
            x = CGSTensor(x_batch, requires_grad=False)
            y = CGSTensor(y_batch)

            if self.use_cgs:
                metrics = self._cgs_train_step(x, y)
            else:
                metrics = self._standard_train_step(x, y)

            total_loss += metrics['loss']
            total_correct += metrics['correct']
            total_samples += metrics['batch_size']
            total_sparsity += metrics.get('sparsity', 0)
            total_gid += metrics.get('avg_gid', 0)
            num_batches += 1

            # Logging
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / num_batches
                acc = total_correct / max(total_samples, 1)
                sparsity = total_sparsity / num_batches if self.use_cgs else 0
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1:4d} | "
                      f"Loss: {avg_loss:.4f} | Acc: {acc:.2%}" +
                      (f" | Sparsity: {sparsity:.1%}" if self.use_cgs else ""))

        return {
            'loss': total_loss / max(num_batches, 1),
            'accuracy': total_correct / max(total_samples, 1),
            'avg_sparsity': total_sparsity / max(num_batches, 1),
            'avg_gid': total_gid / max(num_batches, 1),
            'num_batches': num_batches,
        }

    def _cgs_train_step(self, x: CGSTensor, y: CGSTensor) -> dict:
        """
        One CGS training step — the full pipeline.

        1. Analyze gradients (probe + GID + memory)
        2. Sparsify gradients
        3. Route to parameter blocks
        4. Apply updates
        5. Adapt hyperparameters
        """
        # Step 1: Gradient Intelligence Analysis
        report = self.intelligence.analyze(self.model, self.loss_fn, x, y)
        importance_map = self.intelligence.get_importance_map(report)

        # Extract gradients and GID scores
        gradients = {name: info['gradient'] for name, info in report.items()}
        gid_scores = {name: info['gid_score'] for name, info in report.items()}

        # Step 2: Sparsify gradients
        sparse_grads, sparsity_stats = self.sparsifier.sparsify(gid_scores, gradients)

        # Step 3: Apply sparse gradients to model parameters
        for name, param in self.model.named_parameters():
            if name in sparse_grads:
                param.grad = sparse_grads[name]

        # Step 4: Route updates to parameter blocks
        if self.parameter_blocks:
            self.router.apply_sparse_gradients(
                self.model, importance_map, self.parameter_blocks
            )

        # Step 5: Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Step 6: Adaptive controller step
        avg_gid = float(np.mean(list(gid_scores.values()))) if gid_scores else 0
        gradient_stats = {
            'avg_magnitude': float(np.mean([info['magnitude'] for info in report.values()])),
            'avg_novelty': float(np.mean([info['novelty'] for info in report.values()])),
        }

        controller_update = self.controller.step(
            loss=self._compute_loss(x, y),
            sparsity=sparsity_stats['sparsity'],
            avg_gid=avg_gid,
            gradient_stats=gradient_stats,
        )

        # Apply controller updates
        self.optimizer.set_lr(controller_update['lr'])
        self.sparsifier.set_threshold(controller_update['threshold'])
        self.router.set_threshold(controller_update['threshold'])
        self.intelligence.update_gid_weights(
            controller_update['alpha'],
            controller_update['beta'],
            controller_update['gamma'],
        )

        # Compute batch metrics
        with_predictions = self.model(x)
        predictions = np.argmax(with_predictions.data, axis=1)
        correct = int(np.sum(predictions == y.data.astype(int)))

        return {
            'loss': self._compute_loss(x, y),
            'correct': correct,
            'batch_size': x.shape[0],
            'sparsity': sparsity_stats['sparsity'],
            'avg_gid': avg_gid,
        }

    def _standard_train_step(self, x: CGSTensor, y: CGSTensor) -> dict:
        """Standard training step (no CGS)."""
        self.optimizer.zero_grad()

        # Forward
        output = self.model(x)
        loss = self.loss_fn(output, y)

        # Backward
        loss.backward()

        # Update
        self.optimizer.step()

        # Metrics
        predictions = np.argmax(output.data, axis=1)
        correct = int(np.sum(predictions == y.data.astype(int)))

        return {
            'loss': float(loss.data),
            'correct': correct,
            'batch_size': x.shape[0],
        }

    def _compute_loss(self, x: CGSTensor, y: CGSTensor) -> float:
        """Compute loss without gradient tracking."""
        from ..tensor.tensor import no_grad
        with no_grad():
            output = self.model(x)
            loss = self.loss_fn(output, y)
        return float(loss.data)

    def _validate(self, loader: DataLoader) -> dict:
        """Validate on the provided loader."""
        from ..tensor.tensor import no_grad
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        with no_grad():
            for x_batch, y_batch in loader:
                x = CGSTensor(x_batch)
                y = CGSTensor(y_batch)

                output = self.model(x)
                loss = self.loss_fn(output, y)

                predictions = np.argmax(output.data, axis=1)
                correct = int(np.sum(predictions == y.data.astype(int)))

                total_loss += float(loss.data)
                total_correct += correct
                total_samples += len(x_batch)
                num_batches += 1

        return {
            'loss': total_loss / max(num_batches, 1),
            'accuracy': total_correct / max(total_samples, 1),
        }

    def _print_epoch_summary(self, epoch, total_epochs, train_metrics, val_metrics, epoch_time):
        """Print formatted epoch summary."""
        parts = [f"  📊 Epoch {epoch+1}/{total_epochs}"]
        parts.append(f"Loss: {train_metrics['loss']:.4f}")
        parts.append(f"Acc: {train_metrics['accuracy']:.2%}")
        if val_metrics:
            parts.append(f"Val Loss: {val_metrics['loss']:.4f}")
            parts.append(f"Val Acc: {val_metrics['accuracy']:.2%}")
        if self.use_cgs:
            parts.append(f"Sparsity: {train_metrics.get('avg_sparsity', 0):.1%}")
        parts.append(f"Time: {epoch_time:.1f}s")
        print(" | ".join(parts))

    def get_history(self) -> dict:
        """Return training history."""
        return self._history.copy()

    def get_stats(self) -> dict:
        """Get comprehensive training statistics."""
        stats = {
            'epoch': self._epoch,
            'global_step': self._global_step,
            'history': self._history,
        }
        if self.use_cgs:
            stats.update({
                'intelligence': self.intelligence.get_stats(),
                'sparsifier': self.sparsifier.get_stats(),
                'router': self.router.get_stats(),
                'controller': self.controller.get_stats(),
            })
        return stats
