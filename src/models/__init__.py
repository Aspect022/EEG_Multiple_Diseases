"""ECG Classification Models.

Available model types:
- SNN: Spiking Neural Networks (SpikingResNet)
- Quantum: Hybrid Quantum-Classical CNNs
- Transformer: Swin Transformer variants
"""

from .snn import (
    SpikingResNet,
    SpikingResNet1D,
    create_spiking_resnet,
)
from .quantum import (
    HybridQuantumCNN,
    EfficientHybridCNN,
    create_hybrid_quantum_cnn,
)
from .transformer import (
    SwinECGClassifier,
    create_swin_classifier,
    list_available_models,
)

__all__ = [
    # SNN
    'SpikingResNet',
    'SpikingResNet1D',
    'create_spiking_resnet',
    # Quantum
    'HybridQuantumCNN',
    'EfficientHybridCNN',
    'create_hybrid_quantum_cnn',
    # Transformer
    'SwinECGClassifier',
    'create_swin_classifier',
    'list_available_models',
]
