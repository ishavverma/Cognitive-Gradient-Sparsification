# 🧠 Cognitive Gradient Sparsification (CGS)

**A Self-Sufficient Data-Efficient Learning Framework**

> *CGS is not just an optimization technique — it is a new learning paradigm that redefines how models learn by prioritizing information over volume.*

---

## 🔥 What Is CGS?

CGS is a **gradient-aware adaptive neural system** that learns efficiently from minimal data by selectively updating model parameters using only **high-information gradients**.

Traditional training: `Forward → Loss → Backprop → Update ALL parameters`  
**CGS training**: `Forward → Multi-View Probing → Gradient Intelligence → Selective Update`

### Key Innovation: Gradients Are First-Class Citizens

In CGS, gradients are not a byproduct — they are **modeled explicitly**. The system:
- 🔮 **Thinks before learning** (Gradient Probing Layer)
- 🎯 **Learns selectively** (Gradient Information Density scoring)
- 🧠 **Remembers how it learned** (Gradient Memory Graph)
- ⚡ **Adapts its own learning** (Adaptive Controller)

---

## 🏗️ Architecture: CGS-Net

```
Input Data
    ↓
┌──────────────────────────┐
│   Multi-View Encoder     │  ← Creates 4 diverse views
└──────────────────────────┘
    ↓
┌──────────────────────────┐
│ Representation Space Hub │  ← Attention-based fusion
└──────────────────────────┘
    ↓
┌──────────────────────────┐
│ Gradient Probing Layer   │  ← 🔥 Simulates before learning
└──────────────────────────┘
    ↓
┌──────────────────────────┐
│ Gradient Intelligence    │
│ Engine (GID + Memory)    │  ← 🔥 Scores gradient quality
└──────────────────────────┘
    ↓
┌──────────────────────────┐
│ Sparse Update Router     │  ← 🔥 Dynamic learning pathways
└──────────────────────────┘
    ↓
┌──────────────────────────┐
│ Modular Parameter Blocks │  ← Fine-grained learning control
└──────────────────────────┘
    ↓
Output
```

---

## 🚀 Quick Start

### Installation

```bash
pip install numpy matplotlib
```

### Quick Validation Test

```bash
python experiments/quick_test.py
```

### Train on MNIST

```bash
python experiments/run_mnist.py
```

### Python API

```python
from cgs.model.cgs_net import CGSNet
from cgs.optim.adam import Adam
from cgs.nn.loss import CrossEntropyLoss
from cgs.data.dataset import MNISTDataset
from cgs.data.dataloader import DataLoader
from cgs.training.trainer import CGSTrainer

# Create model
model = CGSNet(input_dim=784, num_classes=10, variant='S')

# Create trainer with CGS enabled
trainer = CGSTrainer(
    model=model,
    optimizer=Adam(model.parameters(), lr=0.001),
    loss_fn=CrossEntropyLoss(),
    use_cgs=True,           # Enable gradient intelligence
    sparsity_threshold=0.3,  # Filter 30%+ of gradients
)

# Train
dataset = MNISTDataset(root='./data/mnist', train=True)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
history = trainer.train(train_loader=loader, epochs=10)
```

---

## 🧩 Core Components

| Component | Module | Description |
|-----------|--------|-------------|
| **Tensor Engine** | `cgs.tensor` | Custom autograd with reverse-mode differentiation |
| **Multi-View Encoder** | `cgs.encoder` | Creates diverse input views for gradient diversity |
| **Gradient Probing** | `cgs.gradient.probing` | "Dry-run" learning simulations |
| **GID Scoring** | `cgs.gradient.gid` | Magnitude × Novelty × Impact scoring |
| **Memory Graph** | `cgs.gradient.memory_graph` | Graph-structured gradient history |
| **Sparse Router** | `cgs.sparse.router` | Dynamic learning pathway selection |
| **Parameter Blocks** | `cgs.sparse.parameter_block` | Independent learning units |
| **Adaptive Controller** | `cgs.controller` | Self-adjusting hyperparameters |
| **Weight Export** | `cgs.export` | Export to PyTorch, TensorFlow, Keras |

---

## 📊 CGS-Net Variants

| Variant | Hidden Dim | Encoder Layers | Param Blocks | Use Case |
|---------|-----------|----------------|--------------|----------|
| **CGS-Net-S** | 64 | 2 | 2 | Small datasets, edge AI |
| **CGS-Net-M** | 128 | 4 | 4 | General tasks |
| **CGS-Net-L** | 256 | 6 | 8 | Scalable systems |

---

## 🔬 Gradient Information Density (GID)

The novel metric at the heart of CGS:

```
GID = α·Magnitude + β·Novelty + γ·Impact

where:
  Magnitude = L2 norm of gradient vector
  Novelty   = Cosine distance from nearest gradient in memory
  Impact    = Predicted change in loss if gradient is applied
```

Only gradients with **high GID** are used for parameter updates.

---

## 🛠️ Implementation Details

- **Language**: Pure Python
- **Dependencies**: NumPy only (for training)
- **No ML frameworks**: All forward pass, backprop, and optimization implemented from scratch
- **Export**: Trained weights can be exported to PyTorch, TensorFlow, Keras

---

## 📁 Project Structure

```
├── cgs/
│   ├── tensor/      # Autograd tensor engine
│   ├── nn/          # Neural network layers
│   ├── optim/       # Optimizers (SGD, Adam)
│   ├── data/        # Datasets and data loading
│   ├── encoder/     # Multi-view encoder
│   ├── gradient/    # ⭐ Gradient intelligence (core innovation)
│   ├── sparse/      # Sparse routing & parameter blocks
│   ├── controller/  # Adaptive hyperparameter control
│   ├── model/       # CGS-Net model assembly
│   ├── training/    # Training pipeline
│   ├── export/      # Weight serialization
│   └── utils/       # Logging, config, visualization
├── experiments/     # Benchmarks and tests
└── config/          # Configuration files
```

---

## 📄 License

Research project — all rights reserved.
