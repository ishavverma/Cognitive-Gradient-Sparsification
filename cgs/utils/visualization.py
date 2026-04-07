"""
Visualization — Training curves, gradient heatmaps.
"""

import numpy as np
import os
from typing import Dict, List, Optional


def plot_training_history(history: dict, save_path: str = 'training_curves.png'):
    """
    Plot training curves (loss, accuracy, sparsity).

    Uses matplotlib if available, otherwise saves raw data.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('CGS Training Results', fontsize=16, fontweight='bold')

        # Loss
        ax = axes[0, 0]
        ax.plot(history.get('train_loss', []), label='Train Loss', color='#2196F3', linewidth=2)
        if history.get('val_loss'):
            ax.plot(history['val_loss'], label='Val Loss', color='#FF9800', linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy
        ax = axes[0, 1]
        ax.plot(history.get('train_acc', []), label='Train Acc', color='#4CAF50', linewidth=2)
        if history.get('val_acc'):
            ax.plot(history['val_acc'], label='Val Acc', color='#E91E63', linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Sparsity
        ax = axes[1, 0]
        if history.get('sparsity'):
            ax.plot(history['sparsity'], color='#9C27B0', linewidth=2)
            ax.fill_between(range(len(history['sparsity'])), history['sparsity'],
                           alpha=0.3, color='#9C27B0')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Sparsity Ratio')
        ax.set_title('Gradient Sparsity')
        ax.grid(True, alpha=0.3)

        # Learning Rate
        ax = axes[1, 1]
        if history.get('lr'):
            ax.plot(history['lr'], color='#FF5722', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📊 Training curves saved to {save_path}")

    except ImportError:
        print("  ⚠️  matplotlib not available. Saving raw data instead.")
        import json
        data_path = save_path.replace('.png', '.json')
        with open(data_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"  📊 Training data saved to {data_path}")


def plot_comparison(results: Dict[str, dict], save_path: str = 'comparison.png'):
    """
    Plot comparison between CGS and standard training.

    Args:
        results: Dict mapping method name → training history.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('CGS vs Standard Training', fontsize=16, fontweight='bold')

        colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

        for i, (name, history) in enumerate(results.items()):
            color = colors[i % len(colors)]

            # Accuracy
            axes[0].plot(history.get('train_acc', []), label=name,
                        color=color, linewidth=2)
            axes[0].set_title('Training Accuracy')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')

            # Loss
            axes[1].plot(history.get('train_loss', []), label=name,
                        color=color, linewidth=2)
            axes[1].set_title('Training Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')

            # Epoch time
            axes[2].plot(history.get('epoch_time', []), label=name,
                        color=color, linewidth=2)
            axes[2].set_title('Epoch Time')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Time (s)')

        for ax in axes:
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📊 Comparison chart saved to {save_path}")

    except ImportError:
        print("  ⚠️  matplotlib not available.")
