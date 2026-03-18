"""
Fusion Models for Multi-Modal Sleep Staging.

Combines 1D raw EEG signals with 2D scalogram representations.
"""

from .fusion_models import (
    EarlyFusionNetwork,
    LateFusionNetwork,
    GatedFusionNetwork,
    MultiModalFeatureExtractor,
    QuantumSNNFusionEarly,
    QuantumSNNFusionGated,
    create_early_fusion,
    create_late_fusion,
    create_gated_fusion,
    create_quantum_snn_fusion_early,
    create_quantum_snn_fusion_gated,
)

__all__ = [
    'EarlyFusionNetwork',
    'LateFusionNetwork',
    'GatedFusionNetwork',
    'MultiModalFeatureExtractor',
    'QuantumSNNFusionEarly',
    'QuantumSNNFusionGated',
    'create_early_fusion',
    'create_late_fusion',
    'create_gated_fusion',
    'create_quantum_snn_fusion_early',
    'create_quantum_snn_fusion_gated',
]
