"""CGS Model — CGS-Net variants."""

from .cgs_net import CGSNet
from .variants import get_variant_config, CGS_NET_S, CGS_NET_M, CGS_NET_L

__all__ = ['CGSNet', 'get_variant_config', 'CGS_NET_S', 'CGS_NET_M', 'CGS_NET_L']
