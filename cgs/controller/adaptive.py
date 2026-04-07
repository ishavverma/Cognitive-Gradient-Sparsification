"""
Adaptive Controller — Dynamic hyperparameter adjustment.

The brain of the CGS training loop's meta-learning:
  - Monitors training metrics (loss, gradient stats, sparsity)
  - Dynamically adjusts sparsity threshold, learning rate, GID weights
  - Implements warm-up, plateau response, and convergence strategies

The model decides: What to learn, When to learn, How much to learn.
"""

import numpy as np
from typing import Optional, Dict


class AdaptiveController:
    """
    Dynamically adjusts CGS training hyperparameters.

    Strategies:
      1. Warm-up: Start dense (low sparsity), increase sparsity over time
      2. Plateau response: Temporarily decrease sparsity when loss stagnates
      3. Convergence detection: Increase sparsity near convergence
      4. Learning rate scheduling: Reduce LR as training progresses

    Args:
        initial_threshold: Starting sparsity threshold.
        initial_lr: Starting learning rate.
        warmup_epochs: Number of warm-up epochs (start dense).
        patience: Steps before plateau detection triggers.
        min_threshold: Minimum sparsity threshold.
        max_threshold: Maximum sparsity threshold.
    """

    def __init__(self, initial_threshold: float = 0.2, initial_lr: float = 0.001,
                 warmup_epochs: int = 3, patience: int = 5,
                 min_threshold: float = 0.05, max_threshold: float = 0.8):
        self.threshold = initial_threshold
        self.initial_threshold = initial_threshold
        self.lr = initial_lr
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # Internal state
        self._step = 0
        self._epoch = 0
        self._best_loss = float('inf')
        self._patience_counter = 0
        self._loss_history: list = []
        self._threshold_history: list = []
        self._lr_history: list = []

        # GID weight scheduling
        self.alpha = 0.3  # magnitude
        self.beta = 0.4   # novelty
        self.gamma = 0.3  # impact

    def step(self, loss: float, sparsity: float, avg_gid: float,
             gradient_stats: Optional[dict] = None) -> dict:
        """
        Update controller state and compute new hyperparameters.

        Called after each training step/batch.

        Args:
            loss: Current training loss.
            sparsity: Current gradient sparsity ratio.
            avg_gid: Average GID score across parameters.
            gradient_stats: Optional additional gradient statistics.

        Returns:
            Dict of updated hyperparameters:
              - 'threshold': new sparsity threshold
              - 'lr': new learning rate
              - 'alpha', 'beta', 'gamma': new GID weights
              - 'action': description of what changed
        """
        self._step += 1
        self._loss_history.append(loss)

        actions = []

        # 1. Warm-up strategy
        if self._epoch < self.warmup_epochs:
            warmup_progress = self._epoch / max(1, self.warmup_epochs)
            self.threshold = self.min_threshold + (self.initial_threshold - self.min_threshold) * warmup_progress
            actions.append('warmup')
        else:
            # 2. Plateau detection
            if loss < self._best_loss - 1e-4:
                self._best_loss = loss
                self._patience_counter = 0
            else:
                self._patience_counter += 1

            if self._patience_counter >= self.patience:
                # Loss is stagnating → temporarily reduce sparsity to allow broader learning
                self.threshold = max(self.min_threshold, self.threshold * 0.8)
                self._patience_counter = 0
                actions.append('plateau_response')

                # Also reduce learning rate
                self.lr = max(self.lr * 0.5, 1e-5)
                actions.append('lr_reduce')

            # 3. Convergence detection: if loss is consistently low, increase sparsity
            if len(self._loss_history) >= 20:
                recent_loss = np.mean(self._loss_history[-20:])
                older_loss = np.mean(self._loss_history[-40:-20]) if len(self._loss_history) >= 40 else recent_loss + 1

                if recent_loss < older_loss * 0.95:  # Loss improving consistently
                    # Gradually increase sparsity (learn less, focus more)
                    self.threshold = min(self.max_threshold, self.threshold * 1.05)
                    actions.append('convergence_sparsify')

        # 4. GID weight adjustment based on gradient characteristics
        if gradient_stats:
            self._adjust_gid_weights(gradient_stats, avg_gid)
            actions.append('gid_weight_adjust')

        # Clamp threshold
        self.threshold = np.clip(self.threshold, self.min_threshold, self.max_threshold)

        # Track history
        self._threshold_history.append(self.threshold)
        self._lr_history.append(self.lr)

        return {
            'threshold': self.threshold,
            'lr': self.lr,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'action': '+'.join(actions) if actions else 'no_change',
        }

    def epoch_step(self, epoch: int, val_loss: Optional[float] = None):
        """Called at the end of each epoch."""
        self._epoch = epoch

        # Learning rate warm-up
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            self.lr = self.initial_lr * warmup_factor

    def _adjust_gid_weights(self, gradient_stats: dict, avg_gid: float):
        """
        Dynamically adjust α, β, γ based on gradient characteristics.

        - If gradients are mostly redundant (low novelty) → increase β (novelty weight)
        - If gradients are very small → increase α (magnitude weight)
        - If predicted impacts are inaccurate → decrease γ (impact weight)
        """
        avg_magnitude = gradient_stats.get('avg_magnitude', 1.0)
        avg_novelty = gradient_stats.get('avg_novelty', 0.5)

        # Low novelty → emphasize novelty more
        if avg_novelty < 0.3:
            self.beta = min(0.6, self.beta + 0.02)
            self.alpha = max(0.15, self.alpha - 0.01)
            self.gamma = max(0.15, self.gamma - 0.01)
        # High novelty → reduce novelty weight, focus on magnitude/impact
        elif avg_novelty > 0.7:
            self.beta = max(0.2, self.beta - 0.02)
            self.alpha = min(0.5, self.alpha + 0.01)
            self.gamma = min(0.5, self.gamma + 0.01)

        # Normalize weights
        total = self.alpha + self.beta + self.gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total

    def get_stats(self) -> dict:
        """Get controller statistics."""
        return {
            'step': self._step,
            'epoch': self._epoch,
            'threshold': self.threshold,
            'lr': self.lr,
            'gid_weights': (self.alpha, self.beta, self.gamma),
            'best_loss': self._best_loss,
            'patience_counter': self._patience_counter,
            'loss_trend': self._loss_history[-10:] if self._loss_history else [],
        }

    def reset(self):
        """Reset controller to initial state."""
        self.threshold = self.initial_threshold
        self.lr = self.initial_lr
        self._step = 0
        self._epoch = 0
        self._best_loss = float('inf')
        self._patience_counter = 0
        self._loss_history.clear()
        self._threshold_history.clear()
        self._lr_history.clear()
