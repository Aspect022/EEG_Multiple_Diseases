"""
Hybrid Quantum-Classical CNN for ECG classification.

Supports GPU-accelerated PennyLane backends (lightning.gpu via cuQuantum),
configurable entanglement topologies, and rotation strategies.

Device fallback: lightning.gpu -> lightning.qubit -> default.qubit
"""

import torch
import torch.nn as nn
import numpy as np
import pennylane as qml
from typing import Optional, List, Tuple
import warnings


# ==========================================================================
# Quantum Device Selection (GPU-aware)
# ==========================================================================

def get_quantum_device(n_qubits: int, diff_method: str = 'best'):
    """
    Create the best available PennyLane device with GPU preference.

    Tries lightning.gpu (NVIDIA cuQuantum) first, then lightning.qubit
    (C++ optimized CPU), then default.qubit (Python fallback).

    Args:
        n_qubits: Number of qubits.
        diff_method: Differentiation method to use.

    Returns:
        Tuple of (device, device_name, diff_method).
    """
    # Priority order of devices to try
    device_configs = [
        ('lightning.gpu', 'adjoint'),
        ('lightning.qubit', 'adjoint'),
        ('default.qubit', 'backprop'),
    ]

    for dev_name, d_method in device_configs:
        try:
            dev = qml.device(dev_name, wires=n_qubits)
            print(f"  [Quantum] Using device: {dev_name} ({d_method})")
            return dev, dev_name, d_method
        except Exception as e:
            continue

    # Should never reach here, but just in case
    dev = qml.device('default.qubit', wires=n_qubits)
    return dev, 'default.qubit', 'backprop'


# ==========================================================================
# Entanglement Strategies
# ==========================================================================

def apply_entanglement(n_qubits: int, entanglement_type: str):
    """
    Apply entangling gates based on the specified topology.

    Args:
        n_qubits: Number of qubits.
        entanglement_type: 'none', 'linear', 'ring', 'full', or 'star'.
    """
    if entanglement_type == 'none':
        return

    elif entanglement_type == 'linear':
        # Chain: q0->q1->q2->q3
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    elif entanglement_type == 'ring':
        # Circular: q0->q1->q2->q3->q0
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

    elif entanglement_type == 'full':
        # All-to-all pairs
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])

    elif entanglement_type == 'star':
        # Hub-spoke: q0 entangled with all others
        for i in range(1, n_qubits):
            qml.CNOT(wires=[0, i])

    else:
        raise ValueError(f"Unknown entanglement: {entanglement_type}")


# ==========================================================================
# Rotation Strategies
# ==========================================================================

# Number of trainable parameters per qubit for each rotation type
ROTATION_PARAMS = {
    'RX': 1,
    'RY': 1,
    'RZ': 1,
    'RXY': 2,
    'RXZ': 2,
    'RYZ': 2,
    'RXYZ': 3,
}

VALID_ROTATIONS = list(ROTATION_PARAMS.keys())
VALID_ENTANGLEMENTS = ['none', 'linear', 'ring', 'full', 'star']


def apply_rotations(weights, qubit: int, rotation_type: str):
    """
    Apply parameterized rotation gates to a single qubit.

    Args:
        weights: Parameter tensor. Shape depends on rotation_type.
        qubit: Qubit index.
        rotation_type: One of 'RX', 'RY', 'RZ', 'RXY', 'RXZ', 'RYZ', 'RXYZ'.
    """
    idx = 0
    if 'X' in rotation_type:
        qml.RX(weights[idx], wires=qubit)
        idx += 1
    if 'Y' in rotation_type:
        qml.RY(weights[idx], wires=qubit)
        idx += 1
    if 'Z' in rotation_type:
        qml.RZ(weights[idx], wires=qubit)
        idx += 1


# ==========================================================================
# Quantum Convolutional Filter (preserved for compatibility)
# ==========================================================================

class QuantumConvLayer(nn.Module):
    """
    Single quantum convolution kernel.

    Encodes a small patch of input into qubit rotations, applies
    variational circuit, and measures expectation values.

    Args:
        n_qubits: Number of qubits. Default: 4.
        n_layers: Variational circuit depth. Default: 2.
        entanglement_type: Entanglement topology. Default: 'ring'.
        rotation_type: Rotation strategy. Default: 'RXYZ'.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        entanglement_type: str = 'ring',
        rotation_type: str = 'RXYZ',
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement_type = entanglement_type
        self.rotation_type = rotation_type

        n_params_per_qubit = ROTATION_PARAMS[rotation_type]

        # Get device
        self.qdev, self.dev_name, self.diff_method = get_quantum_device(n_qubits)

        # Trainable weights
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, n_params_per_qubit) * 0.1
        )

        # Build QNode
        self.qnode = qml.QNode(
            self._circuit,
            self.qdev,
            interface='torch',
            diff_method=self.diff_method,
        )

    def _circuit(self, inputs, weights):
        """Variational quantum circuit."""
        # Angle encoding: always use RY for input encoding
        for i in range(self.n_qubits):
            idx = i % len(inputs)
            qml.RY(inputs[idx] * np.pi, wires=i)

        # Variational layers
        for layer in range(self.n_layers):
            # Parameterized rotations
            for qubit in range(self.n_qubits):
                apply_rotations(weights[layer, qubit], qubit, self.rotation_type)
            # Entanglement
            apply_entanglement(self.n_qubits, self.entanglement_type)

        # Measure
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run quantum circuit on a single input vector.

        Args:
            inputs: (n_qubits,) tensor.
        Returns:
            (n_qubits,) tensor of expectation values.
        """
        result = self.qnode(inputs, self.weights)
        return torch.stack(result)


class QuanvFilter(nn.Module):
    """
    Quantum convolution filter — slides a quantum kernel over spatial features.

    Args:
        in_channels: Input channels. Default: 1.
        out_channels: Output quantum filters. Default: 4.
        kernel_size: Spatial kernel size. Default: 2.
        stride: Convolution stride. Default: 2.
        n_layers: Quantum circuit depth. Default: 2.
        entanglement_type: Topology. Default: 'ring'.
        rotation_type: Rotations. Default: 'RXYZ'.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        n_layers: int = 2,
        entanglement_type: str = 'ring',
        rotation_type: str = 'RXYZ',
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        n_qubits = kernel_size * kernel_size

        self.quantum_layer = QuantumConvLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            entanglement_type=entanglement_type,
            rotation_type=rotation_type,
        )

        # Project quantum outputs to desired channels
        self.channel_proj = nn.Conv2d(n_qubits, out_channels, 1) if out_channels != n_qubits else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, c, h, w = x.shape
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1

        output = torch.zeros(batch_size, self.quantum_layer.n_qubits, out_h, out_w,
                             device=x.device, dtype=x.dtype)

        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, 0,
                           i*self.stride:i*self.stride+self.kernel_size,
                           j*self.stride:j*self.stride+self.kernel_size]
                patch_flat = patch.reshape(batch_size, -1)

                for b in range(batch_size):
                    output[b, :, i, j] = self.quantum_layer(patch_flat[b])

        output = self.channel_proj(output)
        return output


# ==========================================================================
# Hybrid Quantum-Classical CNN
# ==========================================================================

class HybridQuantumCNN(nn.Module):
    """
    Hybrid Quantum-Classical CNN using quantum convolution on preprocessed features.

    Args:
        num_classes: Number of output classes. Default: 5.
        in_channels: Input channels. Default: 3.
        quantum_filters: Quantum conv filters. Default: 4.
        quantum_layers: Circuit depth. Default: 2.
        use_quantum: Enable quantum layer. Default: True.
        entanglement_type: Topology. Default: 'ring'.
        rotation_type: Rotations. Default: 'RXYZ'.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        quantum_filters: int = 4,
        quantum_layers: int = 2,
        use_quantum: bool = True,
        entanglement_type: str = 'ring',
        rotation_type: str = 'RXYZ',
    ):
        super().__init__()
        self.use_quantum = use_quantum

        # Preprocessing: reduce spatial size before quantum layer
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        if use_quantum:
            self.quanv = QuanvFilter(
                in_channels=1, out_channels=quantum_filters,
                kernel_size=2, stride=2, n_layers=quantum_layers,
                entanglement_type=entanglement_type, rotation_type=rotation_type,
            )
            conv1_in = quantum_filters
        else:
            self.quanv = nn.Sequential(
                nn.Conv2d(1, quantum_filters, 2, stride=2),
                nn.BatchNorm2d(quantum_filters),
                nn.ReLU(),
            )
            conv1_in = quantum_filters

        # Classical CNN
        self.conv_layers = nn.Sequential(
            nn.Conv2d(conv1_in, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        x = self.quanv(x)
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


# ==========================================================================
# Efficient Hybrid CNN (Primary quantum model for experiments)
# ==========================================================================

class EfficientHybridCNN(nn.Module):
    """
    Efficient Hybrid Quantum-Classical CNN.

    Classical CNN aggressively downsamples features before a single quantum
    layer processes them. Configurable entanglement and rotation strategies.

    Args:
        num_classes: Output classes. Default: 5.
        in_channels: Input channels. Default: 3.
        n_qubits: Number of qubits. Default: 4.
        n_qlayers: Quantum circuit depth. Default: 2.
        entanglement_type: 'none', 'linear', 'ring', 'full', 'star'. Default: 'ring'.
        rotation_type: 'RX', 'RY', 'RZ', 'RXY', 'RXZ', 'RYZ', 'RXYZ'. Default: 'RXYZ'.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        n_qubits: int = 4,
        n_qlayers: int = 2,
        entanglement_type: str = 'ring',
        rotation_type: str = 'RXYZ',
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.entanglement_type = entanglement_type
        self.rotation_type = rotation_type

        n_params_per_qubit = ROTATION_PARAMS[rotation_type]

        # Classical encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=4, padding=3),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        # Project to n_qubits features
        self.pre_quantum = nn.Linear(128 * 4, n_qubits)

        # Quantum device + circuit
        self.qdev, self.dev_name, self.diff_method = get_quantum_device(n_qubits)
        self.quantum_weights = nn.Parameter(
            torch.randn(n_qlayers, n_qubits, n_params_per_qubit) * 0.1
        )
        self.qnode = qml.QNode(
            self._quantum_layer,
            self.qdev,
            interface='torch',
            diff_method=self.diff_method,
        )

        # Post-quantum classifier
        self.post_quantum = nn.Sequential(
            nn.Linear(n_qubits, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, num_classes),
        )

    def _quantum_layer(self, inputs, weights):
        """Variational quantum layer with configurable entanglement and rotations."""
        # Angle encoding (always RY)
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)

        # Variational layers
        for layer in range(weights.shape[0]):
            for qubit in range(self.n_qubits):
                apply_rotations(weights[layer, qubit], qubit, self.rotation_type)
            apply_entanglement(self.n_qubits, self.entanglement_type)

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Classical encoding
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.pre_quantum(x)
        x = torch.tanh(x) * np.pi  # Scale to [-pi, pi]

        # Quantum processing (batch loop)
        quantum_out = []
        for b in range(batch_size):
            out = self.qnode(x[b], self.quantum_weights)
            quantum_out.append(torch.stack(out))
        x = torch.stack(quantum_out)

        # Classical classifier
        x = self.post_quantum(x)
        return x

    def get_config_name(self) -> str:
        """Return human-readable config name for logging."""
        return f"Q-{self.entanglement_type}-{self.rotation_type}"


# ==========================================================================
# Factory
# ==========================================================================

def create_hybrid_quantum_cnn(
    model_name: str = 'efficient',
    num_classes: int = 5,
    use_quantum: bool = True,
    entanglement_type: str = 'ring',
    rotation_type: str = 'RXYZ',
    **kwargs,
) -> nn.Module:
    """
    Factory function to create Hybrid Quantum-Classical CNN models.

    Args:
        model_name: 'hybrid' (QuanvFilter-based) or 'efficient' (dense quantum layer).
        num_classes: Output classes.
        use_quantum: Whether to use quantum layers (hybrid only).
        entanglement_type: Entanglement topology.
        rotation_type: Rotation strategy.

    Returns:
        Quantum-classical hybrid model.
    """
    if rotation_type not in VALID_ROTATIONS:
        raise ValueError(f"Invalid rotation_type '{rotation_type}'. Must be one of: {VALID_ROTATIONS}")
    if entanglement_type not in VALID_ENTANGLEMENTS:
        raise ValueError(f"Invalid entanglement_type '{entanglement_type}'. Must be one of: {VALID_ENTANGLEMENTS}")

    if model_name == 'hybrid':
        return HybridQuantumCNN(
            num_classes=num_classes, use_quantum=use_quantum,
            entanglement_type=entanglement_type, rotation_type=rotation_type,
            **kwargs
        )
    elif model_name == 'efficient':
        return EfficientHybridCNN(
            num_classes=num_classes,
            entanglement_type=entanglement_type, rotation_type=rotation_type,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
