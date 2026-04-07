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
            print(f"  ⚠️  Config file not found: {path}, using defaults")
            return _default_config()

    with open(path, 'r') as f:
        if path.endswith('.json'):
            return json.load(f)
        else:
            # Simple YAML parser for flat configs
            return _parse_simple_yaml(f.read())


def _parse_simple_yaml(text: str) -> dict:
    """Parse a simple YAML file (flat key-value pairs)."""
    config = {}
    current_section = config

    for line in text.strip().split('\n'):
        line = line.rstrip()
        if not line or line.startswith('#'):
            continue

        # Detect section headers
        if not line.startswith(' ') and line.endswith(':') and ':' not in line[:-1]:
            section = line[:-1].strip()
            config[section] = {}
            current_section = config[section]
            continue

        # Key-value pair
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Type inference
            if value.lower() in ('true', 'yes'):
                value = True
            elif value.lower() in ('false', 'no'):
                value = False
            elif value.replace('.', '').replace('-', '').isdigit():
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass

            current_section[key] = value

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
