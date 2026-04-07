"""
Weight Serializer — Standardized weight tensor serialization.

Saves model parameters in a portable format (.npz) with metadata.
Also provides export functions to PyTorch, TensorFlow, and Keras.
"""

import numpy as np
import os
import json
from typing import Dict, Optional
from ..nn.module import Module


def save_weights(model: Module, path: str, metadata: Optional[dict] = None):
    """
    Save model weights to a .npz file with metadata.

    Args:
        model: The model to save.
        path: File path (will add .npz if not present).
        metadata: Optional metadata dict.
    """
    state = model.state_dict()

    if not path.endswith('.npz'):
        path += '.npz'

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    np.savez(path, **state)

    # Save metadata alongside
    meta = {
        'num_parameters': model.count_parameters(),
        'parameter_names': list(state.keys()),
        'parameter_shapes': {k: list(v.shape) for k, v in state.items()},
    }
    if metadata:
        meta.update(metadata)

    meta_path = path.replace('.npz', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  💾 Saved weights to {path} ({model.count_parameters():,} parameters)")


def load_weights(model: Module, path: str):
    """
    Load model weights from a .npz file.

    Args:
        model: The model to load weights into.
        path: File path to the .npz file.
    """
    if not path.endswith('.npz'):
        path += '.npz'

    data = np.load(path)
    state_dict = {k: data[k] for k in data.files}
    model.load_state_dict(state_dict)
    print(f"  ✅ Loaded weights from {path}")


def export_to_pytorch(model: Module, path: str):
    """
    Export model weights to PyTorch format.

    Requires PyTorch to be installed (only for serialization).

    Args:
        model: The CGS model.
        path: Output .pt file path.
    """
    try:
        import torch

        state = model.state_dict()
        torch_state = {}
        for name, data in state.items():
            torch_state[name] = torch.tensor(data)

        torch.save(torch_state, path)
        print(f"  ✅ Exported to PyTorch format: {path}")

    except ImportError:
        print("  ⚠️  PyTorch not installed. Install with: pip install torch")
        # Fallback: save as numpy
        fallback_path = path.replace('.pt', '_pytorch_weights.npz')
        state = model.state_dict()
        np.savez(fallback_path, **state)
        print(f"  💾 Saved weights as NumPy (load manually into PyTorch): {fallback_path}")


def export_to_tensorflow(model: Module, path: str):
    """
    Export model weights to TensorFlow format.

    Args:
        model: The CGS model.
        path: Output directory for SavedModel or .h5 file.
    """
    try:
        import tensorflow as tf

        state = model.state_dict()

        # Save as .h5 format
        if path.endswith('.h5'):
            # Create variable dict
            variables = {name: tf.Variable(data, name=name)
                        for name, data in state.items()}

            checkpoint = tf.train.Checkpoint(**variables)
            checkpoint.save(path.replace('.h5', ''))
            print(f"  ✅ Exported to TensorFlow checkpoint: {path}")
        else:
            # Save as numpy with TF-compatible naming
            os.makedirs(path, exist_ok=True)
            for name, data in state.items():
                safe_name = name.replace('.', '_')
                np.save(os.path.join(path, f"{safe_name}.npy"), data)
            print(f"  ✅ Exported to TensorFlow format: {path}")

    except ImportError:
        print("  ⚠️  TensorFlow not installed. Install with: pip install tensorflow")
        fallback_path = path + '_tf_weights.npz' if not path.endswith('.npz') else path
        state = model.state_dict()
        np.savez(fallback_path, **state)
        print(f"  💾 Saved weights as NumPy: {fallback_path}")


def export_to_keras(model: Module, path: str):
    """
    Export model weights to Keras format.

    Args:
        model: The CGS model.
        path: Output .keras or .h5 file path.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras

        state = model.state_dict()

        # Save as a weights file
        weights_path = path if path.endswith('.h5') else path + '.h5'
        os.makedirs(os.path.dirname(weights_path) if os.path.dirname(weights_path) else '.', exist_ok=True)

        # Save individual weight arrays
        import h5py
        with h5py.File(weights_path, 'w') as f:
            for name, data in state.items():
                f.create_dataset(name, data=data)
        print(f"  ✅ Exported to Keras format: {weights_path}")

    except ImportError:
        print("  ⚠️  TensorFlow/Keras not installed. Install with: pip install tensorflow")
        fallback_path = path + '_keras_weights.npz' if not path.endswith('.npz') else path
        state = model.state_dict()
        np.savez(fallback_path, **state)
        print(f"  💾 Saved weights as NumPy: {fallback_path}")
