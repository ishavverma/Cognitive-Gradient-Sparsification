"""
CGS-Net model variant configurations.

Three variants:
  - CGS-Net-S: Small — for small datasets and quick experiments
  - CGS-Net-M: Medium — for general tasks
  - CGS-Net-L: Large — for scalable systems
"""


CGS_NET_S = {
    'name': 'CGS-Net-S',
    'encoder_layers': 2,
    'hidden_dim': 64,
    'rep_dim': 64,
    'num_parameter_blocks': 2,
    'dropout': 0.1,
    'noise_std': 0.1,
    'mask_ratio': 0.15,
    'fusion_mode': 'attention',
    'shared_encoder': True,
    'description': 'Small variant for data-scarce environments and quick experiments.',
}

CGS_NET_M = {
    'name': 'CGS-Net-M',
    'encoder_layers': 4,
    'hidden_dim': 128,
    'rep_dim': 128,
    'num_parameter_blocks': 4,
    'dropout': 0.15,
    'noise_std': 0.1,
    'mask_ratio': 0.2,
    'fusion_mode': 'attention',
    'shared_encoder': True,
    'description': 'Medium variant for general classification and learning tasks.',
}

CGS_NET_L = {
    'name': 'CGS-Net-L',
    'encoder_layers': 6,
    'hidden_dim': 256,
    'rep_dim': 256,
    'num_parameter_blocks': 8,
    'dropout': 0.2,
    'noise_std': 0.05,
    'mask_ratio': 0.2,
    'fusion_mode': 'attention',
    'shared_encoder': True,
    'description': 'Large variant for scalable systems and complex tasks.',
}


VARIANTS = {
    'S': CGS_NET_S,
    'M': CGS_NET_M,
    'L': CGS_NET_L,
}


def get_variant_config(variant: str = 'S') -> dict:
    """
    Get configuration for a CGS-Net variant.

    Args:
        variant: One of 'S', 'M', 'L'.

    Returns:
        Configuration dictionary.
    """
    variant = variant.upper()
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(VARIANTS.keys())}")
    return VARIANTS[variant].copy()
