# Cognitive Gradient Sparsification (CGS)

CGS is a NumPy-based research framework for training neural networks with
gradient-aware, adaptive, and sparse updates. It combines multi-view
representations, gradient probing, gradient memory, gradient importance
scoring, sparse routing, and adaptive training controls.

Traditional training updates every parameter after backpropagation:

`Forward -> Loss -> Backpropagation -> Update all parameters`

CGS adds an analysis stage:

`Forward -> Gradient Probe -> GID Scoring -> Sparse Mask -> Selective Update`

## Features

- Custom tensor engine and reverse-mode automatic differentiation
- NumPy neural-network layers, activations, normalization, losses, and optimizers
- Four-view CGS-Net input encoding
- Attention-based representation fusion
- Gradient probing and predicted loss-impact estimation
- Gradient Information Density (GID) scoring
- Gradient novelty tracking with bounded memory graphs
- Threshold, top-K, and hybrid sparsification
- Element-level gradient masking
- Adaptive learning-rate, threshold, and GID-weight control
- Module, layer, and automatic parameter-block routing
- NumPy model serialization and optional PyTorch, TensorFlow, and Keras export
- MNIST benchmark and synthetic smoke tests

## Requirements

- Python 3.8 or newer
- NumPy 1.21 or newer
- Matplotlib 3.5 or newer for charts

Install the core package:

```bash
pip install -e .
```

Install visualization dependencies:

```bash
pip install -e ".[viz]"
```

PyTorch or TensorFlow export can be installed through the optional extras:

```bash
pip install -e ".[export-pytorch]"
pip install -e ".[export-tensorflow]"
```

## Quick Start

Run the framework smoke test:

```bash
python experiments/quick_test.py
```

Run the MNIST benchmark:

```bash
python experiments/run_mnist.py
```

Run the component test suite:

```bash
python -m unittest discover -s tests -v
```

## Python API

```python
from cgs.model.cgs_net import CGSNet
from cgs.optim.adam import Adam
from cgs.nn.loss import CrossEntropyLoss
from cgs.data.dataset import MNISTDataset
from cgs.data.dataloader import DataLoader
from cgs.training.trainer import CGSTrainer

model = CGSNet(input_dim=784, num_classes=10, variant='S')

trainer = CGSTrainer(
    model=model,
    optimizer=Adam(model.parameters(), lr=0.001),
    loss_fn=CrossEntropyLoss(),
    use_cgs=True,
    use_full_probing=False,
    sparsity_mode='hybrid',
    sparsity_threshold=0.3,
    warmup_epochs=2,
)

dataset = MNISTDataset(root='./data/mnist', train=True)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
history = trainer.train(train_loader=loader, epochs=10)
```

Set `use_cgs=False` on `CGSTrainer` to run a standard dense training loop.

## CGS-Net Architecture

```text
Input
  |
  v
Multi-View Encoder
  raw | noisy | masked | scaled
  |
  v
Attention-based Representation Hub
  |
  v
Task Head
  |
  v
Class logits
```

During CGS training, the computed gradients pass through the following stages:

```text
Backpropagation
  |
  v
Gradient preview or quick gradient analysis
  |
  v
GID = alpha * magnitude
    + beta * novelty
    + gamma * impact
  |
  v
Threshold / top-K / hybrid mask
  |
  v
SparseUpdateRouter parameter-block decisions
  |
  v
Optimizer update
  |
  v
AdaptiveController hyperparameter adjustment
```

## CGS-Net Variants

| Variant | Hidden dimension | Encoder layers | Parameter blocks | Intended use |
|---|---:|---:|---:|---|
| CGS-Net-S | 64 | 2 | 2 | Small datasets and quick experiments |
| CGS-Net-M | 128 | 4 | 4 | General classification tasks |
| CGS-Net-L | 256 | 6 | 8 | Larger and more complex tasks |

## Sparsification Modes

The sparsifier supports three parameter-selection and element-masking modes:

- `threshold`: keep values or parameters meeting the GID threshold
- `topk`: keep the configured fraction of the highest-ranked values or parameters
- `hybrid`: require both threshold and top-K criteria

The trainer enables element-level sparsification by default. An element remains
masked unless it passes the selected rule and, for threshold/hybrid modes, the
corresponding parameter GID meets the threshold.

The adaptive controller can change the threshold during training. Warm-up,
loss-plateau response, convergence, and learning-rate scheduling are handled
inside `AdaptiveController`.

## Gradient Intelligence

GID combines:

```text
Magnitude = L2 norm of the gradient
Novelty   = 1 - maximum similarity to stored gradients
Impact    = absolute predicted loss change

GID = alpha * magnitude
    + beta * novelty
    + gamma * impact
```

When full probing is disabled, quick probing uses gradient magnitude as the
impact proxy for faster execution.

Full probing can simulate updates for individual parameters and sample gradient
direction stability. `probe_fraction` controls which parameter groups receive
the more expensive simulation.

## Configuration

The default model, training, CGS, data, and benchmark settings are in
[`config/default.yaml`](config/default.yaml).

The loader is a small built-in YAML parser and also accepts JSON configuration
files. The MNIST benchmark uses the `benchmark` section of this file.

Example:

```yaml
cgs:
  enabled: true
  use_full_probing: false
  gid_alpha: 0.3
  gid_beta: 0.4
  gid_gamma: 0.3
  sparsity_mode: hybrid
  sparsity_threshold: 0.3
  warmup_epochs: 2
  block_granularity: module
```

## Core Modules

| Component | Location | Responsibility |
|---|---|---|
| Tensor engine | `cgs/tensor` | Tensors and reverse-mode autograd |
| Neural-network primitives | `cgs/nn` | Linear, activation, normalization, dropout, and loss modules |
| Optimizers | `cgs/optim` | SGD and Adam |
| Data pipeline | `cgs/data` | MNIST, synthetic data, transforms, and batching |
| Encoders | `cgs/encoder` | Multi-view and representation fusion |
| Gradient intelligence | `cgs/gradient` | Probing, GID, and gradient memory |
| Sparse updates | `cgs/sparse` | Sparsification, routing, and parameter blocks |
| Adaptive control | `cgs/controller` | Dynamic training hyperparameters |
| Training | `cgs/training` | Training loops, callbacks, and metrics |
| Serialization | `cgs/export` | NumPy weights and framework export adapters |
| Configuration and utilities | `cgs/utils` | YAML/JSON loading, logging, seeds, and plotting |

## MNIST Benchmark

[`experiments/run_mnist.py`](experiments/run_mnist.py) can run:

- Standard training on 2 percent of MNIST
- CGS training on 2 percent of MNIST
- CGS training on 0.5 percent of MNIST

It writes:

- Training histories under `logs/`
- Model weights under `models/`
- Epoch checkpoints under `checkpoints/`
- Accuracy and loss charts under `results/`

Benchmark settings, including seed and subset fractions, are configured in
`benchmark` in [`config/default.yaml`](config/default.yaml).

## Testing

The suite includes tests for:

- Nested configuration loading
- Adaptive threshold and learning-rate bounds
- Sparse router constraints
- Element-level and hybrid sparsification
- Weight serialization round trips

The smoke test also validates:

- Analytical versus numerical gradients
- Neural-network forward and backward passes
- CGS-Net construction
- Gradient intelligence
- Synthetic CGS training
- Weight save/load behavior

## Saving and Exporting Models

```python
from cgs.export.serializer import save_weights, load_weights

save_weights(model, './models/cgs_model')
load_weights(model, './models/cgs_model.npz')
```

Optional adapters are available for PyTorch, TensorFlow, and Keras:

```python
from cgs.export.serializer import export_to_pytorch

export_to_pytorch(model, './models/cgs_model.pt')
```

## Project Structure

```text
cgs/
  tensor/        Custom tensor and autograd engine
  nn/            Neural-network primitives
  optim/         SGD and Adam
  data/          Dataset, dataloader, and transforms
  encoder/       Multi-view encoder and representation hub
  gradient/      Probing, GID, memory graph, and intelligence engine
  sparse/        Sparsifier, router, and parameter blocks
  controller/    Adaptive hyperparameter controller
  model/         CGS-Net and variants
  training/      Trainer, callbacks, and metrics
  export/        Weight and framework export
  utils/         Config, seed, logging, and visualization
experiments/     MNIST benchmark and smoke test
config/          Default YAML configuration
tests/           Component unit tests
```

## License

See [`LICENSE.md`](LICENSE.md).
