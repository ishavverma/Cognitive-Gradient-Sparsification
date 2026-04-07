"""CGS Training — Trainer, Callbacks, and Metrics."""

from .trainer import CGSTrainer
from .callbacks import LoggingCallback, CheckpointCallback, EarlyStoppingCallback
from .metrics import Accuracy, TrainingEfficiency, SparsityTracker

__all__ = [
    'CGSTrainer',
    'LoggingCallback', 'CheckpointCallback', 'EarlyStoppingCallback',
    'Accuracy', 'TrainingEfficiency', 'SparsityTracker',
]
