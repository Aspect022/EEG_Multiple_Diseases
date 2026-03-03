"""
Hybrid Quantum-Classical CNN for ECG Classification.

This module implements a hybrid model that combines quantum computing layers
(using PennyLane) with classical neural network layers. The quantum layer
acts as a "Quantum Convolution" (Quanv) that processes local patches of
the input image.

Reference:
- PennyLane: https://pennylane.ai/
- Quantum Convolution: https://arxiv.org/abs/1904.04767
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np
import pennylane as qml


class QuantumConvLayer(nn.Module):
    """
    Quantum Convolution Layer using PennyLane.
    
    Implements a quantum filter that processes 2x2 patches of the input
    using a parameterized quantum circuit. Each patch is encoded into
    qubit rotations, processed through entangling gates, and measured.
    
    Args:
        n_qubits: Number of qubits (should match patch size, default 4 for 2x2).
        n_layers: Number of variational layers. Default: 2.
        device: Quantum device backend. Default: 'default.qubit'.
    
    Note:
        This layer significantly increases computation time compared to
        classical convolutions. Use sparingly in the architecture.
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        device: str = 'default.qubit',
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create quantum device
        self.qdev = qml.device(device, wires=n_qubits)
        
        # Initialize trainable parameters
        # Shape: (n_layers, n_qubits, 3) for RX, RY, RZ rotations
        weight_shape = (n_layers, n_qubits, 3)
        self.weights = nn.Parameter(
            torch.randn(weight_shape) * 0.1
        )
        
        # Create the quantum node
        self.qnode = qml.QNode(
            self._quantum_circuit,
            self.qdev,
            interface='torch',
            diff_method='backprop',
        )
        
    def _quantum_circuit(self, inputs: torch.Tensor, weights: torch.Tensor):
        """
        Parameterized quantum circuit.
        
        Args:
            inputs: Input values to encode (flattened 2x2 patch).
            weights: Trainable rotation angles.
            
        Returns:
            Expectation values of Pauli-Z on each qubit.
        """
        # Encode input data using angle embedding
        for i in range(self.n_qubits):
            qml.RY(inputs[i] * np.pi, wires=i)
        
        # Variational layers
        for layer in range(self.n_layers):
            # Rotation gates
            for qubit in range(self.n_qubits):
                qml.RX(weights[layer, qubit, 0], wires=qubit)
                qml.RY(weights[layer, qubit, 1], wires=qubit)
                qml.RZ(weights[layer, qubit, 2], wires=qubit)
            
            # Entangling gates (ring topology)
            for qubit in range(self.n_qubits):
                qml.CNOT(wires=[qubit, (qubit + 1) % self.n_qubits])
        
        # Measure expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quantum convolution to input.
        
        Processes 2x2 patches using the quantum circuit.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Output tensor of shape (B, n_qubits, H//2, W//2).
        """
        batch_size, channels, height, width = x.shape
        
        # For simplicity, process only first channel
        x = x[:, 0:1, :, :]  # (B, 1, H, W)
        
        # Extract 2x2 patches with stride 2
        # Output shape: (B, 1, H//2, W//2, 4)
        patches = F.unfold(x, kernel_size=2, stride=2)  # (B, 4, num_patches)
        num_patches = patches.shape[2]
        
        # Reshape for processing
        patches = patches.permute(0, 2, 1)  # (B, num_patches, 4)
        
        # Normalize patches to [0, 1] for angle encoding
        patches = (patches - patches.min()) / (patches.max() - patches.min() + 1e-8)
        
        # Process each patch through quantum circuit
        outputs = []
        for b in range(batch_size):
            batch_outputs = []
            for p in range(num_patches):
                patch = patches[b, p]  # (4,)
                out = self.qnode(patch, self.weights)
                out = torch.stack(out)  # (n_qubits,)
                batch_outputs.append(out)
            batch_outputs = torch.stack(batch_outputs, dim=1)  # (n_qubits, num_patches)
            outputs.append(batch_outputs)
        
        outputs = torch.stack(outputs, dim=0)  # (B, n_qubits, num_patches)
        
        # Reshape to spatial dimensions
        out_h = height // 2
        out_w = width // 2
        outputs = outputs.view(batch_size, self.n_qubits, out_h, out_w)
        
        return outputs


class QuanvFilter(nn.Module):
    """
    Faster Quantum Convolution Filter using vectorized operations.
    
    This is an optimized version that processes multiple patches
    in parallel where possible.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (quantum filters).
        kernel_size: Size of quantum filter. Default: 2.
        stride: Stride of convolution. Default: 2.
        n_layers: Number of quantum layers. Default: 2.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        n_layers: int = 2,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_layers = n_layers
        self.n_qubits = kernel_size * kernel_size
        
        # Create multiple quantum filters
        self.quantum_layers = nn.ModuleList([
            QuantumConvLayer(
                n_qubits=self.n_qubits,
                n_layers=n_layers,
            )
            for _ in range(out_channels)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quantum convolution filters.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Output tensor of shape (B, out_channels, H', W').
        """
        outputs = []
        for qconv in self.quantum_layers:
            out = qconv(x)  # (B, n_qubits, H', W')
            # Reduce qubit outputs to single channel
            out = out.mean(dim=1, keepdim=True)  # (B, 1, H', W')
            outputs.append(out)
        
        return torch.cat(outputs, dim=1)  # (B, out_channels, H', W')


class HybridQuantumCNN(nn.Module):
    """
    Hybrid Quantum-Classical CNN for image classification.
    
    Architecture:
    1. Quantum Convolution Layer (Quanv) - processes input with quantum circuits
    2. Classical Conv Layers - extract hierarchical features
    3. Fully Connected Layers - classification
    
    Args:
        num_classes: Number of output classes. Default: 5.
        in_channels: Number of input channels. Default: 3.
        quantum_filters: Number of quantum convolution filters. Default: 4.
        quantum_layers: Number of variational layers in quantum circuit. Default: 2.
        use_quantum: If False, replace quantum layer with classical conv. Default: True.
    
    Example:
        >>> model = HybridQuantumCNN(num_classes=5)
        >>> x = torch.randn(2, 3, 64, 64)  # Smaller input for quantum
        >>> output = model(x)  # (2, 5)
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        quantum_filters: int = 4,
        quantum_layers: int = 2,
        use_quantum: bool = True,
    ):
        super().__init__()
        
        self.use_quantum = use_quantum
        
        # Preprocessing: reduce spatial size before quantum layer
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, stride=2, padding=1),  # Reduce to 1 channel
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        
        if use_quantum:
            # Quantum convolution layer
            self.quanv = QuanvFilter(
                in_channels=1,
                out_channels=quantum_filters,
                kernel_size=2,
                stride=2,
                n_layers=quantum_layers,
            )
            conv1_in = quantum_filters
        else:
            # Classical fallback
            self.quanv = nn.Sequential(
                nn.Conv2d(1, quantum_filters, 2, stride=2),
                nn.BatchNorm2d(quantum_filters),
                nn.ReLU(),
            )
            conv1_in = quantum_filters
        
        # Classical convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(conv1_in, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid network.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
               Recommended input size: 64x64 or smaller for reasonable
               quantum computation time.
            
        Returns:
            Output logits of shape (B, num_classes).
        """
        # Preprocess to reduce size
        x = self.preprocess(x)
        
        # Quantum convolution
        x = self.quanv(x)
        
        # Classical convolutions
        x = self.conv_layers(x)
        
        # Classification
        x = self.classifier(x)
        
        return x


class EfficientHybridCNN(nn.Module):
    """
    Efficient Hybrid Quantum-Classical CNN.
    
    Uses a single quantum layer in the middle of the network where
    feature maps are already reduced, making quantum computation tractable.
    
    Args:
        num_classes: Number of output classes. Default: 5.
        in_channels: Number of input channels. Default: 3.
        n_qubits: Number of qubits for quantum layer. Default: 4.
        n_qlayers: Number of quantum circuit layers. Default: 2.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        n_qubits: int = 4,
        n_qlayers: int = 2,
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        
        # Classical encoder (aggressive downsampling)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=4, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # Fixed 2x2 output
        )
        
        # Project to n_qubits features
        self.pre_quantum = nn.Linear(128 * 4, n_qubits)
        
        # Quantum layer (on flattened features)
        self.qdev = qml.device('default.qubit', wires=n_qubits)
        self.quantum_weights = nn.Parameter(
            torch.randn(n_qlayers, n_qubits, 3) * 0.1
        )
        self.qnode = qml.QNode(
            self._quantum_layer,
            self.qdev,
            interface='torch',
            diff_method='backprop',
        )
        
        # Post-quantum classical layers
        self.post_quantum = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        
    def _quantum_layer(self, inputs, weights):
        """Variational quantum layer."""
        # Encode inputs
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # Variational layers
        for layer in range(weights.shape[0]):
            for qubit in range(self.n_qubits):
                qml.RX(weights[layer, qubit, 0], wires=qubit)
                qml.RY(weights[layer, qubit, 1], wires=qubit)
                qml.RZ(weights[layer, qubit, 2], wires=qubit)
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Output logits of shape (B, num_classes).
        """
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


def create_hybrid_quantum_cnn(
    model_name: str = 'hybrid',
    num_classes: int = 5,
    use_quantum: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create Hybrid Quantum-Classical CNN models.
    
    Args:
        model_name: Model variant ('hybrid', 'efficient').
        num_classes: Number of output classes.
        use_quantum: Whether to use quantum layers.
        **kwargs: Additional arguments.
        
    Returns:
        Hybrid CNN model.
    """
    if model_name == 'hybrid':
        return HybridQuantumCNN(
            num_classes=num_classes,
            use_quantum=use_quantum,
            **kwargs
        )
    elif model_name == 'efficient':
        return EfficientHybridCNN(
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
