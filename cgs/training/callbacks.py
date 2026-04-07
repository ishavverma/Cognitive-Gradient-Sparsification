"""
Training Callbacks — Logging, Checkpointing, Early Stopping.
"""

import numpy as np
import os
import json
from typing import Optional, Dict


class Callback:
    """Base callback class."""

    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_end(self, trainer, epoch: int, train_metrics: dict,
                     val_metrics: dict) -> bool:
        """Return True to stop training."""
        return False


class LoggingCallback(Callback):
    """Structured logging to file."""

    def __init__(self, log_dir: str = './logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'training_log.jsonl')

    def on_train_begin(self, trainer):
        with open(self.log_file, 'w') as f:
            f.write(json.dumps({'event': 'train_begin',
                                'params': trainer.model.count_parameters()}) + '\n')

    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        entry = {
            'event': 'epoch_end',
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        return False

    def on_train_end(self, trainer):
        # Save final history
        history_path = os.path.join(self.log_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(trainer.get_history(), f, indent=2)


class CheckpointCallback(Callback):
    """Save model checkpoints."""

    def __init__(self, save_dir: str = './checkpoints', save_best: bool = True,
                 monitor: str = 'val_loss'):
        self.save_dir = save_dir
        self.save_best = save_best
        self.monitor = monitor
        self.best_value = float('inf') if 'loss' in monitor else 0
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        # Determine current monitored value
        if self.monitor == 'val_loss':
            current = val_metrics.get('loss', float('inf'))
            is_better = current < self.best_value
        elif self.monitor == 'val_acc':
            current = val_metrics.get('accuracy', 0)
            is_better = current > self.best_value
        else:
            current = train_metrics.get('loss', float('inf'))
            is_better = current < self.best_value

        if self.save_best and is_better:
            self.best_value = current
            path = os.path.join(self.save_dir, 'best_model.npz')
            state = trainer.model.state_dict()
            np.savez(path, **state)
            print(f"  💾 Saved best model (epoch {epoch+1}, {self.monitor}={current:.4f})")

        # Also save periodic checkpoints
        if (epoch + 1) % 5 == 0:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.npz')
            state = trainer.model.state_dict()
            np.savez(path, **state)

        return False


class EarlyStoppingCallback(Callback):
    """Stop training if no improvement for `patience` epochs."""

    def __init__(self, patience: int = 10, monitor: str = 'val_loss',
                 min_delta: float = 1e-4):
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float('inf') if 'loss' in monitor else 0

    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        if self.monitor == 'val_loss':
            current = val_metrics.get('loss', float('inf'))
            is_better = current < self.best_value - self.min_delta
        elif self.monitor == 'val_acc':
            current = val_metrics.get('accuracy', 0)
            is_better = current > self.best_value + self.min_delta
        else:
            current = train_metrics.get('loss', float('inf'))
            is_better = current < self.best_value - self.min_delta

        if is_better:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
