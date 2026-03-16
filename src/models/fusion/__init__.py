"""Fusion model architectures for EEG sleep staging."""

from .gated_fusion import GatedFusionModule, MultiStreamFusion, ClassificationHead
from .fusion_a import FusionA, create_fusion_a
from .fusion_b import FusionB, create_fusion_b
from .fusion_c import FusionC, create_fusion_c

__all__ = [
    'GatedFusionModule',
    'MultiStreamFusion',
    'ClassificationHead',
    'FusionA',
    'FusionB',
    'FusionC',
    'create_fusion_a',
    'create_fusion_b',
    'create_fusion_c',
]
