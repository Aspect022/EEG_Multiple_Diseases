"""Quantum models for EEG Sleep Staging."""

from .hybrid_cnn import (
    EfficientHybridCNN,
    create_hybrid_quantum_cnn,
    VALID_ROTATIONS,
    VALID_ENTANGLEMENTS,
)

from .quantum_circuit import (
    VectorizedQuantumCircuit,
    QuantumMeasurement,
    QuantumGates,
    ROTATION_MAP,
    ENTANGLEMENT_MAP,
)

__all__ = [
    'EfficientHybridCNN',
    'create_hybrid_quantum_cnn',
    'VectorizedQuantumCircuit',
    'QuantumMeasurement',
    'QuantumGates',
    'VALID_ROTATIONS',
    'VALID_ENTANGLEMENTS',
    'ROTATION_MAP',
    'ENTANGLEMENT_MAP',
]
