#!/usr/bin/env python3
"""
TCANet: Temporal Convolutional Attention Network for EEG Classification.

A clean, modular PyTorch implementation combining:
- Multi-Scale CNN (MSCNet) for spatial-spectral feature extraction
- Temporal Convolutional Network (TCN) for temporal dependencies
- Quantum Layer (placeholder for future quantum circuits)
- Transformer Attention for global temporal relationships

Author: EEG Research Team
Date: March 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================================
# MODULE 1: MSCNet (Multi-Scale CNN)
# ============================================================================

class MSCNet(nn.Module):
    """
    Multi-Scale Convolutional Network for EEG feature extraction.
    
    Extracts features at three different temporal scales using parallel
    convolutional branches, then concatenates them for rich representation.
    
    Architecture:
        Input: (batch, 1, channels, time)
        ├─ Branch 1: Conv2D(kernel=125) → DepthwiseConv → BN → ELU → Pool → Dropout
        ├─ Branch 2: Conv2D(kernel=62)  → DepthwiseConv → BN → ELU → Pool → Dropout
        └─ Branch 3: Conv2D(kernel=31)  → DepthwiseConv → BN → ELU → Pool → Dropout
            ↓
        Concatenate → Reshape → Output: (batch, time_reduced, features)
    
    Args:
        channels: Number of EEG channels (default: 22)
        f1: Number of filters in first layer (default: 40)
        pooling_size: Temporal pooling size (default: 75)
        dropout: Dropout rate (default: 0.5)
    
    Example:
        >>> msc = MSCNet(channels=22, f1=40)
        >>> x = torch.randn(64, 1, 22, 1000)
        >>> out = msc(x)
        >>> print(out.shape)  # (64, 13, 120)
    """
    
    def __init__(
        self,
        channels: int = 22,
        f1: int = 40,
        pooling_size: int = 75,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.channels = channels
        self.f1 = f1
        self.pooling_size = pooling_size
        
        # Three parallel branches with different kernel sizes
        # Hook: Could add more branches for finer scale analysis
        self.branch1 = self._make_branch(channels, f1, kernel_size=125, dropout=dropout)
        self.branch2 = self._make_branch(channels, f1, kernel_size=62, dropout=dropout)
        self.branch3 = self._make_branch(channels, f1, kernel_size=31, dropout=dropout)
        
        # Total features = f1 * 3 branches
        self.total_features = f1 * 3
        
        self._initialize_weights()
    
    def _make_branch(
        self,
        channels: int,
        f1: int,
        kernel_size: int,
        dropout: float,
    ) -> nn.Sequential:
        """
        Create a single MSC branch.
        
        Args:
            channels: Input EEG channels
            f1: Number of filters
            kernel_size: Temporal kernel size
            dropout: Dropout rate
        
        Returns:
            Sequential branch module
        """
        padding = kernel_size // 2  # 'same' padding
        
        return nn.Sequential(
            # Conv2D: (batch, 1, channels, time) → (batch, f1, channels, time)
            nn.Conv2d(1, f1, kernel_size=(1, kernel_size), padding=(0, padding), bias=False),
            
            # Depthwise Conv2D: (batch, f1, channels, time) → (batch, f1, 1, time)
            # Each filter operates on single channel
            nn.Conv2d(f1, f1, kernel_size=(channels, 1), groups=f1, bias=False),
            
            # BatchNorm2D
            nn.BatchNorm2d(f1),
            
            # ELU activation
            nn.ELU(),
            
            # Average Pooling: reduce temporal dimension
            nn.AvgPool2d(kernel_size=(1, self.pooling_size)),
            
            # Dropout
            nn.Dropout(dropout),
        )
    
    def _initialize_weights(self):
        """Initialize convolutional weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale branches.
        
        Args:
            x: Input tensor of shape (batch, 1, channels, time)
        
        Returns:
            Feature tensor of shape (batch, time_reduced, features)
            where features = f1 * 3, time_reduced = time // pooling_size
        """
        # Process through each branch
        out1 = self.branch1(x)  # (batch, f1, 1, time//pool)
        out2 = self.branch2(x)  # (batch, f1, 1, time//pool)
        out3 = self.branch3(x)  # (batch, f1, 1, time//pool)
        
        # Concatenate along channel dimension
        # (batch, f1*3, 1, time//pool)
        concatenated = torch.cat([out1, out2, out3], dim=1)
        
        # Reshape: (batch, time//pool, f1*3)
        batch_size = concatenated.size(0)
        time_reduced = concatenated.size(3)
        
        # Flatten and transpose: (batch, f1*3, 1, time//pool) → (batch, time//pool, f1*3)
        out = concatenated.view(batch_size, -1, time_reduced)
        out = out.transpose(1, 2)
        
        return out


# ============================================================================
# MODULE 2: TCN (Temporal Convolutional Network)
# ============================================================================

class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution with left-side padding only.
    
    Ensures no future information leakage by padding only on the left side.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        dilation: Dilation factor (default: 1)
    
    Example:
        >>> conv = CausalConv1d(64, 64, kernel_size=3, dilation=2)
        >>> x = torch.randn(32, 64, 100)
        >>> out = conv(x)
        >>> print(out.shape)  # (32, 64, 100)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Calculate padding for causal convolution
        # padding = (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation
        
        # Actual convolution (no built-in padding)
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            dilation=dilation,
            bias=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with left-side padding.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Output tensor of shape (batch, channels, time)
        """
        # Pad only on left side: (batch, channels, padding + time)
        x_padded = F.pad(x, (self.padding, 0))
        
        # Convolve: (batch, channels, time)
        out = self.conv(x_padded)
        
        return out


class TCNBlock(nn.Module):
    """
    Single TCN Block with causal convolution, normalization, and residual.
    
    Architecture:
        Input → CausalConv1d → BatchNorm1d → ELU → Dropout → Residual → Output
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size (default: 3)
        dilation: Dilation factor (default: 1)
        dropout: Dropout rate (default: 0.2)
    
    Example:
        >>> block = TCNBlock(64, 64, kernel_size=3, dilation=2)
        >>> x = torch.randn(32, 64, 100)
        >>> out = block(x)
        >>> print(out.shape)  # (32, 64, 100)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Causal convolution
        self.conv = CausalConv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            dilation=dilation,
        )
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(out_channels)
        
        # Activation
        self.activation = nn.ELU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv if dimensions don't match)
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Output tensor of shape (batch, out_channels, time)
        """
        # Main path
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection
        residual = self.residual(x)
        
        # Add and return
        out = out + residual
        out = self.activation(out)  # Final activation
        
        return out


class TCN(nn.Module):
    """
    Temporal Convolutional Network with stacked dilated causal convolutions.
    
    Architecture:
        Input: (batch, time, features)
            ↓ (transpose)
        (batch, features, time)
            ↓
        TCN Block 1: dilation=1
        TCN Block 2: dilation=2
        TCN Block 3: dilation=4
        ... (depth layers with dilation=2^i)
            ↓ (transpose)
        Output: (batch, time, filters)
    
    Args:
        input_dim: Input feature dimension (default: 120)
        filters: Number of filters (default: 64)
        kernel_size: Convolution kernel size (default: 3)
        depth: Number of stacked layers (default: 4)
        dropout: Dropout rate (default: 0.2)
    
    Example:
        >>> tcn = TCN(input_dim=120, filters=64, depth=4)
        >>> x = torch.randn(32, 13, 120)
        >>> out = tcn(x)
        >>> print(out.shape)  # (32, 13, 64)
    """
    
    def __init__(
        self,
        input_dim: int = 120,
        filters: int = 64,
        kernel_size: int = 3,
        depth: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.filters = filters
        self.depth = depth
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, filters, kernel_size=1, bias=False)
        
        # Stacked TCN blocks with increasing dilation
        # Hook: Could make depth configurable per application
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** i
            self.blocks.append(
                TCNBlock(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN layers.
        
        Args:
            x: Input tensor of shape (batch, time, features)
        
        Returns:
            Output tensor of shape (batch, time, filters)
        """
        # Transpose: (batch, time, features) → (batch, features, time)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # Pass through TCN blocks
        for block in self.blocks:
            x = block(x)
        
        # Transpose back: (batch, filters, time) → (batch, time, filters)
        x = x.transpose(1, 2)
        
        return x


# ============================================================================
# MODULE 3: Quantum Layer (Placeholder)
# ============================================================================

class QuantumLayer(nn.Module):
    """
    Quantum Layer placeholder for future quantum circuit integration.
    
    Currently implements a simple linear transformation, but designed to be
    replaced with actual quantum circuits from Pennylane or Qiskit.
    
    Architecture (Current):
        Input: (batch, time, features)
            ↓
        Linear(features, features)
            ↓
        Output: (batch, time, features)
    
    Future Architecture (Planned):
        Input: (batch, time, features)
            ↓
        Classical → Quantum encoding (n_qubits)
            ↓
        Variational Quantum Circuit (layers)
            ↓
        Measurement → Classical output
            ↓
        Output: (batch, time, features)
    
    Args:
        features: Feature dimension (default: 64)
        n_qubits: Number of quantum bits (default: 8)
    
    Example:
        >>> quantum = QuantumLayer(features=64)
        >>> x = torch.randn(32, 13, 64)
        >>> out = quantum(x)
        >>> print(out.shape)  # (32, 13, 64)
    
    TODO:
        - Replace with Pennylane qnode for actual quantum computation
        - Add quantum encoding layer (angle encoding or amplitude encoding)
        - Implement variational quantum circuit with trainable parameters
        - Add measurement layer (Pauli-Z expectation values)
    """
    
    def __init__(self, features: int = 64, n_qubits: int = 8):
        super().__init__()
        self.features = features
        self.n_qubits = n_qubits
        
        # Placeholder: Simple linear transformation
        # TODO: Replace with actual quantum circuit
        # Example Pennylane integration:
        #   import pennylane as qml
        #   dev = qml.device('default.qubit', wires=n_qubits)
        #   @qml.qnode(dev)
        #   def quantum_circuit(inputs, weights):
        #       qml.AngleEmbedding(inputs, wires=range(n_qubits))
        #       qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        #       return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        self.linear = nn.Linear(features, features)
        
        # Hook for future quantum weights
        # self.quantum_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (currently classical placeholder).
        
        Args:
            x: Input tensor of shape (batch, time, features)
        
        Returns:
            Output tensor of shape (batch, time, features)
        """
        # Apply linear transformation (placeholder for quantum circuit)
        out = self.linear(x)
        
        # TODO: Replace with actual quantum circuit
        # For Pennylane:
        #   batch_size, time, features = x.shape
        #   x_flat = x.view(-1, features)
        #   quantum_out = quantum_circuit(x_flat, self.quantum_weights)
        #   out = quantum_out.view(batch_size, time, -1)
        
        return out


# ============================================================================
# MODULE 4: Transformer Attention
# ============================================================================

class TransformerAttention(nn.Module):
    """
    Multi-Head Self-Attention for global temporal relationships.
    
    Captures long-range dependencies in the temporal dimension using
    multi-head attention mechanism.
    
    Architecture:
        Input: (batch, time, features)
            ↓
        Multi-Head Attention (batch_first=True)
            ↓
        Dropout + Residual
            ↓
        LayerNorm
            ↓
        Output: (batch, time, features)
    
    Note:
        No feed-forward block (kept minimal as per original TCANet paper)
    
    Args:
        features: Feature dimension (default: 64)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
    
    Example:
        >>> attn = TransformerAttention(features=64, num_heads=8)
        >>> x = torch.randn(32, 13, 64)
        >>> out = attn(x)
        >>> print(out.shape)  # (32, 13, 64)
    
    Hook:
        - Could add positional encoding before attention
        - Could extend to full transformer encoder with feed-forward
        - Could add layer normalization before attention (Pre-LN)
    """
    
    def __init__(
        self,
        features: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.features = features
        self.num_heads = num_heads
        
        # Multi-head attention (batch_first for convenience)
        self.attention = nn.MultiheadAttention(
            embed_dim=features,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization (applied after residual)
        self.layer_norm = nn.LayerNorm(features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through self-attention.
        
        Args:
            x: Input tensor of shape (batch, time, features)
        
        Returns:
            Output tensor of shape (batch, time, features)
        """
        # Self-attention
        # attn_output: (batch, time, features)
        attn_output, _ = self.attention(x, x, x, need_weights=False)
        
        # Dropout
        attn_output = self.dropout(attn_output)
        
        # Residual connection
        out = x + attn_output
        
        # Layer normalization
        out = self.layer_norm(out)
        
        return out


# ============================================================================
# FINAL MODEL: TCANetClean
# ============================================================================

class TCANetClean(nn.Module):
    """
    TCANet: Temporal Convolutional Attention Network for EEG Classification.
    
    Complete architecture combining multi-scale CNN, TCN, quantum layer,
    and transformer attention for robust EEG feature learning.
    
    Architecture:
        Input: (batch, channels, time)
            ↓ (unsqueeze)
        (batch, 1, channels, time)
            ↓
        MSCNet → (batch, time_reduced, features)  [features = f1 * 3]
            ↓
        TCN → (batch, time_reduced, filters)
            ↓
        QuantumLayer → (batch, time_reduced, filters)
            ↓
        TransformerAttention → (batch, time_reduced, filters)
            ↓
        Flatten → (batch, time_reduced * filters)
            ↓
        Classifier → (batch, num_classes)
    
    Args:
        channels: Number of EEG channels (default: 22)
        time: Number of time samples (default: 1000)
        num_classes: Number of output classes (default: 4)
        f1: MSCNet filters (default: 40)
        filters: TCN filters (default: 64)
        pooling_size: MSCNet pooling size (default: 75)
        tcn_depth: TCN block depth (default: 4)
        num_heads: Transformer attention heads (default: 8)
        dropout: Dropout rate (default: 0.5)
    
    Example:
        >>> model = TCANetClean(channels=22, time=1000, num_classes=4)
        >>> x = torch.randn(64, 22, 1000)
        >>> logits, features = model(x)
        >>> print(logits.shape)  # (64, 4)
        >>> print(features.shape)  # (64, 832)
    
    Input/Output:
        Input: (batch, channels, time) - raw EEG signal
        Output: 
            - logits: (batch, num_classes) - classification scores
            - features: (batch, time_reduced * filters) - learned features
    
    Compatibility:
        - Works with nn.CrossEntropyLoss
        - Supports mixed precision training
        - GPU/CPU agnostic
    """
    
    def __init__(
        self,
        channels: int = 22,
        time: int = 1000,
        num_classes: int = 4,
        f1: int = 40,
        filters: int = 64,
        pooling_size: int = 75,
        tcn_depth: int = 4,
        num_heads: int = 8,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.channels = channels
        self.time = time
        self.num_classes = num_classes
        self.f1 = f1
        self.filters = filters
        self.pooling_size = pooling_size
        
        # Calculate reduced time dimension after pooling
        self.time_reduced = time // pooling_size
        
        # Module 1: Multi-Scale CNN
        self.mscnet = MSCNet(
            channels=channels,
            f1=f1,
            pooling_size=pooling_size,
            dropout=dropout,
        )
        
        # Module 2: Temporal Convolutional Network
        self.tcn = TCN(
            input_dim=f1 * 3,  # MSCNet output features
            filters=filters,
            depth=tcn_depth,
            dropout=dropout,
        )
        
        # Module 3: Quantum Layer (placeholder)
        self.quantum = QuantumLayer(features=filters)
        
        # Module 4: Transformer Attention
        self.attention = TransformerAttention(
            features=filters,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Classifier
        # Input size: filters * time_reduced
        self.classifier_input_size = filters * self.time_reduced
        self.classifier = nn.Linear(self.classifier_input_size, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through complete TCANet architecture.
        
        Args:
            x: Input EEG tensor of shape (batch, channels, time)
               Example: (64, 22, 1000) for 64 samples, 22 channels, 1000 time points
        
        Returns:
            Tuple of:
                - logits: Classification logits of shape (batch, num_classes)
                - features: Learned features of shape (batch, time_reduced * filters)
        
        Example:
            >>> model = TCANetClean()
            >>> x = torch.randn(8, 22, 1000)
            >>> logits, features = model(x)
            >>> print(logits.shape)   # (8, 4)
            >>> print(features.shape) # (8, 832)
        """
        batch_size = x.size(0)
        
        # Step 1: Add channel dimension
        # (batch, channels, time) → (batch, 1, channels, time)
        x = x.unsqueeze(1)
        
        # Step 2: Multi-Scale CNN
        # (batch, 1, channels, time) → (batch, time_reduced, f1*3)
        x = self.mscnet(x)
        
        # Step 3: Temporal Convolutional Network
        # (batch, time_reduced, f1*3) → (batch, time_reduced, filters)
        x = self.tcn(x)
        
        # Step 4: Quantum Layer (placeholder)
        # (batch, time_reduced, filters) → (batch, time_reduced, filters)
        x = self.quantum(x)
        
        # Step 5: Transformer Attention
        # (batch, time_reduced, filters) → (batch, time_reduced, filters)
        x = self.attention(x)
        
        # Step 6: Flatten
        # (batch, time_reduced, filters) → (batch, time_reduced * filters)
        features = x.view(batch_size, -1)
        
        # Step 7: Classifier
        # (batch, time_reduced * filters) → (batch, num_classes)
        logits = self.classifier(features)
        
        return logits, features


# ============================================================================
# TEST BLOCK
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TCANetClean - Model Test")
    print("=" * 70)
    
    # Create model with default parameters
    model = TCANetClean(
        channels=22,
        time=1000,
        num_classes=4,
        f1=40,
        filters=64,
        pooling_size=75,
        tcn_depth=4,
        num_heads=8,
        dropout=0.5,
    )
    
    # Print model architecture
    print(f"\nModel Architecture:")
    print(f"  Input: (batch, 22, 1000)")
    print(f"  MSCNet filters: {40 * 3}")
    print(f"  TCN filters: 64")
    print(f"  Time reduced: {1000 // 75}")
    print(f"  Classifier input: {64 * (1000 // 75)}")
    print(f"  Output: (batch, 4)")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create dummy input
    batch_size = 8
    x = torch.randn(batch_size, 22, 1000)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, features = model(x)
    
    # Print output shapes
    print(f"\nOutput shape: {logits.shape}")
    print(f"Feature shape: {features.shape}")
    
    # Verify expected shapes
    assert logits.shape == (batch_size, 4), f"Expected (8, 4), got {logits.shape}"
    assert features.shape == (batch_size, 64 * (1000 // 75)), f"Feature shape mismatch"
    
    print("\n✓ All tests passed!")
    print("=" * 70)
