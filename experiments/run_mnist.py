"""
Run MNIST Benchmark — CGS vs Standard Training.

Demonstrates CGS-Net's data-efficient learning on MNIST.
Trains with both CGS-enabled and standard training for comparison.

NOTE: Pure numpy training is slower than PyTorch/TF. We use a 10% subset
for practical training times while still demonstrating the CGS paradigm.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from cgs.utils.seeding import set_seed
from cgs.tensor.tensor import CGSTensor
from cgs.nn.loss import CrossEntropyLoss
from cgs.optim.adam import Adam
from cgs.data.dataset import MNISTDataset
from cgs.data.dataloader import DataLoader
from cgs.model.cgs_net import CGSNet
from cgs.training.trainer import CGSTrainer
from cgs.training.callbacks import LoggingCallback, CheckpointCallback
from cgs.export.serializer import save_weights
from cgs.utils.visualization import plot_training_history, plot_comparison


def run_experiment(use_cgs: bool, subset_fraction: float = 0.1,
                   test_subset: float = 0.2,
                   epochs: int = 5, variant: str = 'S',
                   batch_size: int = 64, lr: float = 0.001):
    """Run a single training experiment."""
    set_seed(42)

    label = "CGS" if use_cgs else "Standard"
    print(f"\n{'='*60}")
    print(f"  Experiment: {label} Training")
    print(f"  Data fraction: {subset_fraction:.0%}")
    print(f"  Model variant: CGS-Net-{variant}")
    print(f"{'='*60}")

    # Load data
    print("\n  Loading MNIST dataset...")
    train_dataset = MNISTDataset(root='./data/mnist', train=True,
                                  subset_fraction=subset_fraction)
    test_dataset = MNISTDataset(root='./data/mnist', train=False,
                                 subset_fraction=test_subset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Test samples:  {len(test_dataset):,}")

    # Create model
    model = CGSNet(input_dim=784, num_classes=10, variant=variant)
    print(f"\n  Model: {model.config['name']} ({model.count_parameters():,} parameters)")

    # Create optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    # Create trainer
    os.makedirs(f'./logs/{label.lower()}', exist_ok=True)
    os.makedirs(f'./checkpoints/{label.lower()}', exist_ok=True)

    callbacks = [
        LoggingCallback(log_dir=f'./logs/{label.lower()}'),
        CheckpointCallback(save_dir=f'./checkpoints/{label.lower()}'),
    ]

    trainer = CGSTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=CrossEntropyLoss(),
        use_cgs=use_cgs,
        use_full_probing=False,  # Quick mode for faster training
        sparsity_threshold=0.3,
        warmup_epochs=1,
        callbacks=callbacks,
    )

    # Train
    history = trainer.train(
        train_loader=train_loader,
        epochs=epochs,
        val_loader=test_loader,
        log_interval=20,
    )

    # Save model
    os.makedirs('./models', exist_ok=True)
    save_weights(model, f'./models/{label.lower()}_model')

    return history


def main():
    os.makedirs('./results', exist_ok=True)

    print("="*60)
    print("  CGS — MNIST Benchmark (CGS vs Standard Training)")
    print("="*60)

    total_start = time.time()
    results = {}

    # 1. Standard training (baseline) — 2% of MNIST
    print("\n" + "~"*60)
    print("  [1/3] Standard Training Baseline (2% MNIST)")
    print("~"*60)
    results['Standard (2%)'] = run_experiment(
        use_cgs=False, subset_fraction=0.02, test_subset=0.1,
        epochs=3, batch_size=32, lr=0.001
    )

    # 2. CGS training — same 2% of MNIST 
    print("\n" + "~"*60)
    print("  [2/3] CGS Training (2% MNIST)")
    print("~"*60)
    results['CGS (2%)'] = run_experiment(
        use_cgs=True, subset_fraction=0.02, test_subset=0.1,
        epochs=3, batch_size=32, lr=0.001
    )

    # 3. CGS with very limited data (0.5%) — data-efficiency stress test
    print("\n" + "~"*60)
    print("  [3/3] CGS Data-Efficiency Test (0.5% MNIST)")
    print("~"*60)
    results['CGS (0.5%)'] = run_experiment(
        use_cgs=True, subset_fraction=0.005, test_subset=0.1,
        epochs=5, batch_size=16, lr=0.002
    )

    total_time = time.time() - total_start

    # Plot comparison
    plot_comparison(results, './results/comparison.png')

    # Plot individual training curves
    for name, history in results.items():
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
        plot_training_history(history, f'./results/{safe_name}_training.png')

    # Print final comparison table
    print("\n" + "="*60)
    print("  FINAL RESULTS — CGS vs Standard Training")
    print("="*60)
    print(f"\n  {'Method':<20} {'Train Acc':>10} {'Val Acc':>10} {'Avg Time':>10} {'Sparsity':>10}")
    print(f"  {'-'*60}")
    for name, history in results.items():
        train_acc = history['train_acc'][-1] if history['train_acc'] else 0
        val_acc = history['val_acc'][-1] if history['val_acc'] else 0
        avg_time = np.mean(history['epoch_time']) if history['epoch_time'] else 0
        sparsity = np.mean(history['sparsity']) if history.get('sparsity') and history['sparsity'] else 0
        print(f"  {name:<20} {train_acc:>9.2%} {val_acc:>9.2%} {avg_time:>9.1f}s {sparsity:>9.1%}")

    print(f"\n  Total benchmark time: {total_time:.1f}s")
    print(f"  Results saved to ./results/")
    print("="*60)


if __name__ == '__main__':
    main()
