"""
Backend Abstraction Layer — CuPy/NumPy unified interface.

Provides a seamless backend switcher so that the entire CGS framework
can run on either CPU (NumPy) or GPU (CuPy) without code changes.

Usage:
    from cgs.backend import xp, to_device, get_array_module, GPU_AVAILABLE

    # xp is the default array module (np or cp)
    data = xp.zeros((3, 4))

    # Move data to/from GPU
    gpu_data = to_device(data, 'cuda')
    cpu_data = to_device(gpu_data, 'cpu')

    # Auto-detect module from existing array
    arr_mod = get_array_module(some_array)  # returns np or cp
"""

import numpy as np
from typing import Union

# ──────────────────── GPU Detection ────────────────────

try:
    import cupy as cp
    GPU_AVAILABLE = True
    _default_device = 'cpu'  # Still default to CPU; user opts in
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    _default_device = 'cpu'


# ──────────────────── Array Module Selection ────────────────────

def get_array_module(data) -> 'module':
    """
    Return the appropriate array module (numpy or cupy) for the given data.

    Args:
        data: A numpy or cupy ndarray.

    Returns:
        numpy or cupy module.
    """
    if GPU_AVAILABLE and isinstance(data, cp.ndarray):
        return cp
    return np


def get_default_module() -> 'module':
    """Return the default array module based on current device setting."""
    if _default_device == 'cuda' and GPU_AVAILABLE:
        return cp
    return np


# Convenience alias — most code should use this
xp = np  # Updated at runtime by set_default_device()


# ──────────────────── Device Management ────────────────────

def set_default_device(device: str):
    """
    Set the default compute device for new tensors.

    Args:
        device: 'cpu' or 'cuda' (or 'gpu').
    """
    global _default_device, xp
    device = _normalize_device(device)

    if device == 'cuda' and not GPU_AVAILABLE:
        raise RuntimeError(
            "CUDA requested but CuPy is not installed. "
            "Install with: pip install cupy-cuda12x  (match your CUDA version)"
        )

    _default_device = device
    xp = cp if device == 'cuda' else np


def get_default_device() -> str:
    """Get the current default device."""
    return _default_device


def _normalize_device(device: str) -> str:
    """Normalize device string to 'cpu' or 'cuda'."""
    device = device.lower().strip()
    if device in ('cuda', 'gpu', 'cupy'):
        return 'cuda'
    return 'cpu'


# ──────────────────── Data Transfer ────────────────────

def to_device(data, device: str = 'cpu'):
    """
    Transfer array data to the specified device.

    Args:
        data: numpy.ndarray, cupy.ndarray, list, float, or int.
        device: Target device ('cpu' or 'cuda').

    Returns:
        Array on the target device.
    """
    device = _normalize_device(device)

    if device == 'cuda':
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not installed. Cannot move data to GPU.")
        if isinstance(data, np.ndarray):
            return cp.asarray(data)
        elif GPU_AVAILABLE and isinstance(data, cp.ndarray):
            return data  # Already on GPU
        else:
            return cp.asarray(np.asarray(data))

    else:  # cpu
        if GPU_AVAILABLE and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.asarray(data)


def to_numpy(data) -> np.ndarray:
    """
    Ensure data is a numpy array (always on CPU).

    Args:
        data: Any array-like.

    Returns:
        numpy.ndarray on CPU.
    """
    if GPU_AVAILABLE and isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(data)


def is_gpu_array(data) -> bool:
    """Check if data is a CuPy (GPU) array."""
    return GPU_AVAILABLE and isinstance(data, cp.ndarray)


# ──────────────────── Default Dtype ────────────────────

DEFAULT_DTYPE = np.float32  # Changed from float64 for 2x speed/memory


def get_default_dtype():
    """Get the default floating-point dtype."""
    return DEFAULT_DTYPE


def set_default_dtype(dtype):
    """Set the default floating-point dtype (np.float32 or np.float64)."""
    global DEFAULT_DTYPE
    DEFAULT_DTYPE = dtype


# ──────────────────── GPU Info ────────────────────

def gpu_info() -> dict:
    """Get GPU device information (if available)."""
    if not GPU_AVAILABLE:
        return {'available': False, 'reason': 'CuPy not installed'}

    try:
        device = cp.cuda.Device()
        mem_info = device.mem_info
        return {
            'available': True,
            'device_id': device.id,
            'name': cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode(),
            'total_memory_mb': mem_info[1] / (1024 ** 2),
            'free_memory_mb': mem_info[0] / (1024 ** 2),
            'used_memory_mb': (mem_info[1] - mem_info[0]) / (1024 ** 2),
        }
    except Exception as e:
        return {'available': True, 'error': str(e)}
