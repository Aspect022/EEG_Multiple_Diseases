"""
Hybrid Quantum-Classical CNN for EEG Sleep Staging.

Uses a pure-PyTorch vectorized quantum circuit (no PennyLane).
All quantum operations are batched and GPU-native.

Architecture:
    1. Classical encoder: Conv2d layers downsample 224×224 → feature vector
    2. Pre-quantum projection: features → n_qubits angles in [-π, π]
    3. Quantum circuit: Amplitude encoding → variational layers → measurement
    4. Post-quantum classifier: Pauli-Z expectations → class logits

Supports:
    - 7 rotation types: RX, RY, RZ, RXY, RXZ, RYZ, RXYZ
    - 2 entanglement types: ring (circular), full (all-to-all)
    - GPU batching (no per-sample loop)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from .quantum_circuit import (
    VectorizedQuantumCircuit,
    QuantumMeasurement,
    ROTATION_MAP,
    ENTANGLEMENT_MAP,
)


# Valid rotation and entanglement types (pipeline-facing names)
VALID_ROTATIONS = list(ROTATION_MAP.keys())
VALID_ENTANGLEMENTS = list(ENTANGLEMENT_MAP.keys())


# ==========================================================================
# Efficient Hybrid CNN (Primary quantum model)
# ==========================================================================

class EfficientHybridCNN(nn.Module):
    """
    Efficient Hybrid Quantum-Classical CNN.

    Classical CNN downsamples the scalogram, then a vectorized quantum
    circuit processes the compressed features. All operations are
    batched — no per-sample loops.

    Args:
        num_classes: Output classes (default: 5 for sleep staging)
        in_channels: Input channels (default: 3 for RGB scalograms)
        n_qubits: Number of qubits (default: 8)
        n_qlayers: Quantum circuit depth (default: 3)
        entanglement_type: 'ring' or 'full'
        rotation_type: 'RX', 'RY', 'RZ', 'RXY', 'RXZ', 'RYZ', 'RXYZ'
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        n_qubits: int = 8,
        n_qlayers: int = 3,
        entanglement_type: str = 'ring',
        rotation_type: str = 'RXYZ',
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.entanglement_type = entanglement_type
        self.rotation_type = rotation_type

        # Map pipeline names to circuit names
        circuit_rotation = ROTATION_MAP.get(rotation_type, rotation_type.lower())
        circuit_entanglement = ENTANGLEMENT_MAP.get(entanglement_type, entanglement_type)

        # ===== Classical Encoder =====
        # Downsample 224×224 → compact feature vector
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=4, padding=3),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        # ===== Pre-Quantum Projection =====
        # 128 * 2 * 2 = 512 → n_qubits
        self.pre_quantum = nn.Linear(128 * 4, n_qubits)

        # ===== Vectorized Quantum Circuit =====
        self.circuit = VectorizedQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_qlayers,
            rotation_axes=circuit_rotation,
            entanglement=circuit_entanglement,
        )

        # ===== Measurement =====
        self.measurement = QuantumMeasurement(n_qubits=n_qubits)

        # ===== Post-Quantum Classifier =====
        self.post_quantum = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, 3, 224, 224) scalogram images
        Returns:
            (batch, num_classes) class logits
        """
        batch_size = x.shape[0]

        # Classical encoding
        x = self.encoder(x)                      # (B, 128, 2, 2)
        x = x.view(batch_size, -1)               # (B, 512)
        x = self.pre_quantum(x)                   # (B, n_qubits)
        x = torch.tanh(x) * np.pi                # Scale to [-π, π]

        # Quantum processing (BATCHED — no per-sample loop!)
        quantum_state = self.circuit(x)           # (B, 2^n_qubits)
        quantum_features = self.measurement(quantum_state)  # (B, n_qubits)

        # Ensure real-valued output for classifier
        quantum_features = quantum_features.float()

        # Classical classifier
        logits = self.post_quantum(quantum_features)  # (B, num_classes)
        return logits

    def get_config_name(self) -> str:
        """Return human-readable config name for logging."""
        return f"Q-{self.entanglement_type}-{self.rotation_type}"


# ==========================================================================
# Factory
# ==========================================================================

def create_hybrid_quantum_cnn(
    model_name: str = 'efficient',
    num_classes: int = 5,
    entanglement_type: str = 'ring',
    rotation_type: str = 'RXYZ',
    **kwargs,
) -> nn.Module:
    """
    Factory function to create Hybrid Quantum-Classical CNN.

    Args:
        model_name: Only 'efficient' is supported now.
        num_classes: Output classes.
        entanglement_type: 'ring' or 'full'.
        rotation_type: 'RX', 'RY', 'RZ', 'RXY', 'RXZ', 'RYZ', 'RXYZ'.

    Returns:
        EfficientHybridCNN model.
    """
    if rotation_type not in VALID_ROTATIONS:
        raise ValueError(
            f"Invalid rotation_type '{rotation_type}'. "
            f"Must be one of: {VALID_ROTATIONS}"
        )
    if entanglement_type not in VALID_ENTANGLEMENTS:
        raise ValueError(
            f"Invalid entanglement_type '{entanglement_type}'. "
            f"Must be one of: {VALID_ENTANGLEMENTS}"
        )

    return EfficientHybridCNN(
        num_classes=num_classes,
        entanglement_type=entanglement_type,
        rotation_type=rotation_type,
        **kwargs,
    )
