"""SNN 1D models for raw EEG signal processing."""

from .snn_classifier import (
    SNN1D,
    create_snn_1d_lif,
    create_snn_1d_attention,
)
from .spiking_vit import (
    SpikingViT1D,
    SpikingViT1DConfig,
    create_spiking_vit_1d,
)
from .lif_neuron import LIFNeuron, LIFLayer
from .attention import MultiScaleSpikingAttention, GlobalSpikingAttention

__all__ = [
    'SNN1D',
    'create_snn_1d_lif',
    'create_snn_1d_attention',
    'SpikingViT1D',
    'SpikingViT1DConfig',
    'create_spiking_vit_1d',
    'LIFNeuron',
    'LIFLayer',
    'MultiScaleSpikingAttention',
    'GlobalSpikingAttention',
]
