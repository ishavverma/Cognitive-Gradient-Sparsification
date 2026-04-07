"""
Quick Test — Validates the CGS framework end-to-end with synthetic data.

This is a fast smoke test that verifies:
  1. Tensor autograd works correctly
  2. Neural network layers work
  3. CGS-Net model builds successfully 
  4. CGS training pipeline runs
  5. Weight serialization works

Run this to verify installation: python experiments/quick_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from cgs.utils.seeding import set_seed


def test_autograd():
    """Test automatic differentiation correctness."""
    print("\n  🔍 Test 1: Autograd Correctness")

    from cgs.tensor.tensor import CGSTensor
    from cgs.tensor.autograd import numerical_gradient
    from cgs.tensor.ops import matmul, relu, tensor_sum
    from cgs.tensor.functional import cross_entropy

    # Test 1: Simple gradient
    x = CGSTensor(np.random.randn(3, 4), requires_grad=True)
    w = CGSTensor(np.random.randn(4, 2), requires_grad=True)

    def fn(w_):
        y = matmul(CGSTensor(x.data), w_)
        return tensor_sum(y)

    # Analytical gradient
    y = matmul(x, w)
    loss = tensor_sum(y)
    loss.backward()
    analytical_grad = w.grad.copy()

    # Numerical gradient
    w.zero_grad()
    num_grad = numerical_gradient(fn, w)

    error = np.max(np.abs(analytical_grad - num_grad))
    status = "✅ PASS" if error < 1e-5 else "❌ FAIL"
    print(f"     Gradient error: {error:.2e} — {status}")

    # Test 2: Cross-entropy gradient
    logits = CGSTensor(np.random.randn(4, 3), requires_grad=True)
    targets = CGSTensor(np.array([0, 1, 2, 1]))

    def ce_fn(l):
        return cross_entropy(l, CGSTensor(targets.data))

    loss = cross_entropy(logits, targets)
    loss.backward()
    analytical = logits.grad.copy()

    logits.zero_grad()
    numerical = numerical_gradient(ce_fn, logits)

    error = np.max(np.abs(analytical - numerical))
    status = "✅ PASS" if error < 1e-4 else "❌ FAIL"
    print(f"     Cross-entropy gradient error: {error:.2e} — {status}")

    return error < 1e-4


def test_neural_network():
    """Test neural network layers."""
    print("\n  🔍 Test 2: Neural Network Layers")

    from cgs.tensor.tensor import CGSTensor
    from cgs.nn.linear import Linear
    from cgs.nn.activation import ReLU, GELU
    from cgs.nn.normalization import LayerNorm
    from cgs.nn.container import Sequential
    from cgs.nn.loss import CrossEntropyLoss

    # Build a simple model
    model = Sequential(
        Linear(10, 32),
        ReLU(),
        LayerNorm(32),
        Linear(32, 5),
    )

    print(f"     Model: {model.count_parameters()} parameters")

    # Forward pass
    x = CGSTensor(np.random.randn(8, 10))
    output = model(x)
    print(f"     Input shape: {x.shape} → Output shape: {output.shape}")

    # Loss and backward
    targets = CGSTensor(np.array([0, 1, 2, 3, 4, 0, 1, 2]))
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(output, targets)
    loss.backward()

    # Check gradients exist
    has_grads = all(p.grad is not None for p in model.parameters())
    status = "✅ PASS" if has_grads and output.shape == (8, 5) else "❌ FAIL"
    print(f"     Loss: {float(loss.data):.4f}, Gradients present: {has_grads} — {status}")

    return has_grads


def test_cgs_net():
    """Test CGS-Net model construction."""
    print("\n  🔍 Test 3: CGS-Net Model")

    from cgs.tensor.tensor import CGSTensor
    from cgs.model.cgs_net import CGSNet
    from cgs.nn.loss import CrossEntropyLoss

    # Build CGS-Net-S
    model = CGSNet(input_dim=20, num_classes=5, variant='S')
    print(f"     {model}")

    # Forward pass
    x = CGSTensor(np.random.randn(4, 20))
    output = model(x)
    print(f"     Input: {x.shape} → Output: {output.shape}")

    # Backward
    targets = CGSTensor(np.array([0, 1, 2, 3]))
    loss = CrossEntropyLoss()(output, targets)
    loss.backward()

    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_count = sum(1 for _ in model.parameters())
    status = "✅ PASS" if output.shape == (4, 5) else "❌ FAIL"
    print(f"     Loss: {float(loss.data):.4f}, Grads: {grad_count}/{total_count} — {status}")

    return output.shape == (4, 5)


def test_gradient_intelligence():
    """Test Gradient Intelligence Engine."""
    print("\n  🔍 Test 4: Gradient Intelligence Engine")

    from cgs.tensor.tensor import CGSTensor
    from cgs.model.cgs_net import CGSNet
    from cgs.nn.loss import CrossEntropyLoss
    from cgs.gradient.intelligence import GradientIntelligenceEngine

    model = CGSNet(input_dim=20, num_classes=5, variant='S')
    engine = GradientIntelligenceEngine(use_full_probing=False)

    x = CGSTensor(np.random.randn(4, 20))
    y = CGSTensor(np.array([0, 1, 2, 3]))
    loss_fn = CrossEntropyLoss()

    report = engine.analyze(model, loss_fn, x, y)
    importance_map = engine.get_importance_map(report)

    print(f"     Parameters analyzed: {len(report)}")
    print(f"     GID scores: {[f'{v:.3f}' for v in list(importance_map.values())[:5]]}")

    has_scores = len(importance_map) > 0
    status = "✅ PASS" if has_scores else "❌ FAIL"
    print(f"     Memory graph nodes: {engine.memory_graph.get_stats()['total_nodes']} — {status}")

    return has_scores


def test_sparse_training():
    """Test sparse training with CGS pipeline."""
    print("\n  🔍 Test 5: CGS Training Pipeline (Synthetic Data)")

    from cgs.tensor.tensor import CGSTensor
    from cgs.model.cgs_net import CGSNet
    from cgs.nn.loss import CrossEntropyLoss
    from cgs.optim.adam import Adam
    from cgs.data.dataset import SyntheticDataset
    from cgs.data.dataloader import DataLoader
    from cgs.training.trainer import CGSTrainer

    # Synthetic data
    dataset = SyntheticDataset(num_samples=200, input_dim=20, num_classes=5)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = CGSNet(input_dim=20, num_classes=5, variant='S')
    optimizer = Adam(model.parameters(), lr=0.005)

    # Train with CGS
    trainer = CGSTrainer(
        model=model, optimizer=optimizer,
        loss_fn=CrossEntropyLoss(),
        use_cgs=True, use_full_probing=False,
        warmup_epochs=1,
    )

    history = trainer.train(train_loader=loader, epochs=3, log_interval=10)

    # Check training improved
    initial_loss = history['train_loss'][0]
    final_loss = history['train_loss'][-1]
    improved = final_loss < initial_loss

    status = "✅ PASS" if improved else "⚠️  MARGINAL"
    print(f"\n     Initial loss: {initial_loss:.4f} → Final loss: {final_loss:.4f}")
    print(f"     Loss decreased: {improved} — {status}")

    return improved


def test_serialization():
    """Test weight save/load."""
    print("\n  🔍 Test 6: Weight Serialization")

    from cgs.tensor.tensor import CGSTensor
    from cgs.model.cgs_net import CGSNet
    from cgs.export.serializer import save_weights, load_weights

    model = CGSNet(input_dim=20, num_classes=5, variant='S')
    model.eval()  # Disable stochastic layers for deterministic output
    x = CGSTensor(np.random.randn(2, 20))
    out1 = model(x).data.copy()

    # Save
    os.makedirs('./tmp_test', exist_ok=True)
    save_weights(model, './tmp_test/test_model')

    # Load into new model
    model2 = CGSNet(input_dim=20, num_classes=5, variant='S')
    model2.eval()  # Must also be in eval mode
    load_weights(model2, './tmp_test/test_model')

    out2 = model2(x).data.copy()
    match = np.allclose(out1, out2, atol=1e-6)

    status = "✅ PASS" if match else "❌ FAIL"
    print(f"     Outputs match after save/load: {match} — {status}")

    # Cleanup
    import shutil
    shutil.rmtree('./tmp_test', ignore_errors=True)

    return match


def main():
    set_seed(42)

    print("="*60)
    print("  🧠 CGS Framework — Quick Validation Test")
    print("="*60)

    results = {
        'Autograd': test_autograd(),
        'Neural Network': test_neural_network(),
        'CGS-Net Model': test_cgs_net(),
        'Gradient Intelligence': test_gradient_intelligence(),
        'CGS Training': test_sparse_training(),
        'Serialization': test_serialization(),
    }

    print("\n" + "="*60)
    print("  📋 Test Summary")
    print("="*60)

    all_pass = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print(f"\n  {'🎉 All tests passed!' if all_pass else '⚠️  Some tests failed.'}")
    print("="*60)
    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
