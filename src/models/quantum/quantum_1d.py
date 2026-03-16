"""
Quantum model for raw 1D EEG signals.

Ported from ECG project's QuantumPath. Uses:
  1. Conv1d Feature Compressor: (6, 3000) → 64 compressed features
  2. Quantum Encoding Layer: 64 → 8 angles in [0, π]
  3. VectorizedQuantumCircuit: batched quantum simulation
  4. Pauli-Z Measurement: 8 quantum features
  5. Classifier: 8 → 5 classes

Supports all 14 rotation × entanglement configurations.

Input:  (batch, 6, 3000)
Output: (batch, 5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from .quantum_circuit import VectorizedQuantumCircuit, QuantumMeasurement
from .quantum_circuit import ROTATION_MAP, ENTANGLEMENT_MAP


# ──────────────────────── Feature Compressor ────────────────────────

class EEGFeatureCompressor(nn.Module):
    """
    Compress raw EEG to quantum-encodable feature vector.
    
    Pipeline:
        (6, 3000) → Conv1d(6→64, k=5, s=5) → BN → ELU
                   → Conv1d(64→64, k=5, s=5) → BN → ELU
                   → MaxPool(k=4) → Flatten → Linear → 64 features
    
    Uses standard Conv1d (no quantization for best accuracy).
    """
    
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 64,
        output_features: int = 64,
    ):
        super().__init__()
        
        # Two-stage downsampling
        self.conv1 = nn.Conv1d(
            in_channels, hidden_channels,
            kernel_size=5, stride=5, padding=2, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = nn.Conv1d(
            hidden_channels, hidden_channels,
            kernel_size=5, stride=5, padding=2, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        # MaxPool for further compression
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # 3000 // 5 = 600, 600 // 5 = 120, 120 // 4 = 30
        self.fc = nn.Linear(hidden_channels * 30, output_features)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 6, 3000)
        Returns:
            (batch, 64)
        """
        x = F.elu(self.bn1(self.conv1(x)))   # (batch, 64, 600)
        x = F.elu(self.bn2(self.conv2(x)))   # (batch, 64, 120)
        x = self.pool(x)                      # (batch, 64, 30)
        x = x.flatten(1)                      # (batch, 1920)
        x = self.dropout(self.fc(x))          # (batch, 64)
        return x


# ──────────────────────── Quantum Encoding ────────────────────────

class QuantumEncodingLayer(nn.Module):
    """
    Encode classical features into quantum-ready angles.
    
    Projects features to qubit dimension and normalizes to [0, π].
    """
    
    def __init__(self, feature_dim: int = 64, n_qubits: int = 8):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ELU(),
            nn.Linear(32, n_qubits),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, feature_dim)
        Returns:
            (batch, n_qubits) — angles in [0, π]
        """
        x = self.projection(x)
        return torch.sigmoid(x) * np.pi


# ──────────────────────── Quantum 1D Model ────────────────────────

class Quantum1DEEG(nn.Module):
    """
    Quantum model for raw 1D EEG signals.
    
    Full pipeline:
        Raw EEG (6, 3000) → Feature Compression → Quantum Encoding →
        Variational Circuit → Measurement → Classifier → 5 classes
    
    Args:
        in_channels: Number of EEG channels (default 6)
        num_classes: Number of output classes (default 5)
        n_qubits: Number of qubits (default 8)
        n_layers: Number of variational layers (default 3)
        rotation: Rotation type ('RX', 'RY', ..., 'RXYZ')
        entanglement: Entanglement type ('ring' or 'full')
    """
    
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 5,
        n_qubits: int = 8,
        n_layers: int = 3,
        rotation: str = 'RY',
        entanglement: str = 'ring',
        feature_dim: int = 64,
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.rotation = rotation
        self.entanglement = entanglement
        
        # Map rotation/entanglement names to circuit format
        rotation_axes = ROTATION_MAP.get(rotation, ['y'])
        entanglement_type = ENTANGLEMENT_MAP.get(entanglement, 'circular')
        
        # Feature compression
        self.compressor = EEGFeatureCompressor(
            in_channels=in_channels,
            output_features=feature_dim,
        )
        
        # Quantum encoding
        self.encoder = QuantumEncodingLayer(
            feature_dim=feature_dim,
            n_qubits=n_qubits,
        )
        
        # Variational quantum circuit
        self.circuit = VectorizedQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_axes=rotation_axes,
            entanglement=entanglement_type,
        )
        
        # Measurement
        self.measurement = QuantumMeasurement(n_qubits=n_qubits)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 6, 3000) — raw EEG signal
        Returns:
            (batch, num_classes) logits
        """
        # Handle channel mismatch
        if x.shape[1] < 6:
            pad = torch.zeros(
                x.shape[0], 6 - x.shape[1], x.shape[2],
                device=x.device, dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)
        
        # Compress features
        compressed = self.compressor(x)          # (batch, 64)
        
        # Encode to quantum angles
        angles = self.encoder(compressed)         # (batch, n_qubits)
        
        # Quantum circuit
        quantum_state = self.circuit(angles)      # (batch, 2^n_qubits)
        
        # Measure
        quantum_features = self.measurement(quantum_state)  # (batch, n_qubits)
        
        # Classify
        logits = self.classifier(quantum_features)
        
        return logits
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────── Factory ────────────────────────

def create_quantum_1d(
    num_classes: int = 5,
    in_channels: int = 6,
    rotation: str = 'RY',
    entanglement: str = 'ring',
    n_qubits: int = 8,
    n_layers: int = 3,
) -> Quantum1DEEG:
    """Create a Quantum-1D model with specified configuration."""
    return Quantum1DEEG(
        in_channels=in_channels,
        num_classes=num_classes,
        n_qubits=n_qubits,
        n_layers=n_layers,
        rotation=rotation,
        entanglement=entanglement,
    )
