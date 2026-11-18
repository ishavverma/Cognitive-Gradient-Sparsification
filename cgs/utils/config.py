"""YAML Configuration Loader."""

import os
import json
from typing import Optional


def load_config(path: str = 'config/default.yaml') -> dict:
    """
    Load configuration from a YAML or JSON file.

    Uses JSON parser for YAML-subset configs to avoid
    external dependency on PyYAML.
    """
    if not os.path.exists(path):
        # Try JSON variant
        json_path = path.replace('.yaml', '.json').replace('.yml', '.json')
        if os.path.exists(json_path):
            path = json_path
        else:
            print(f"  [warning] Config file not found: {path}, using defaults")
            return _default_config()

    with open(path, 'r') as f:
        if path.endswith('.json'):
            return json.load(f)
        else:
            # Simple YAML parser for flat configs
            return _parse_simple_yaml(f.read())


def _parse_simple_yaml(text: str) -> dict:
    """Parse a small YAML subset including nested benchmark sections."""
    config = {}
    stack = [( -1, config)]

    for raw_line in text.strip().split('\n'):
        if not raw_line.strip() or raw_line.lstrip().startswith('#'):
            continue

        indentation = len(raw_line) - len(raw_line.lstrip(' '))
        line = raw_line.strip()
        if ':' not in line:
            continue

        key, raw_value = line.split(':', 1)
        key = key.strip()
        raw_value = raw_value.strip()

        while stack and indentation <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if not raw_value:
            child = {}
            parent[key] = child
            stack.append((indentation, child))
            continue

        value = raw_value
        if value.lower() in ('true', 'yes'):
            value = True
        elif value.lower() in ('false', 'no'):
            value = False
        else:
            try:
                value = float(value) if '.' in value else int(value)
            except ValueError:
                value = value.strip("'\"")
        parent[key] = value

    return config


def _default_config() -> dict:
    """Return default configuration."""
    return {
        'model': {
            'variant': 'S',
            'input_dim': 784,
            'num_classes': 10,
        },
        'training': {
            'epochs': 10,
            'batch_size': 64,
            'learning_rate': 0.001,
            'optimizer': 'adam',
        },
        'cgs': {
            'enabled': True,
            'use_full_probing': False,
            'gid_alpha': 0.3,
            'gid_beta': 0.4,
            'gid_gamma': 0.3,
            'sparsity_mode': 'hybrid',
            'sparsity_threshold': 0.3,
            'warmup_epochs': 2,
        },
        'data': {
            'dataset': 'mnist',
            'subset_fraction': 1.0,
            'normalize': True,
        },
    }
