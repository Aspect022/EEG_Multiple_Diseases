"""
Conditional Routing Fusion: SNN (fast) → Confidence Gate → Quantum (slow).

Inspired by ECG project's dual-path architecture (D:/Projects/ECG/src/models/v2).

Architecture:
    1. Fast Path:  Raw EEG → SNN-1D-Attention → features (128-d)
    2. Gate:       Confidence estimator on SNN features → c ∈ [0, 1]
    3. Slow Path:  Raw EEG → Quantum-1D-Ring-RYZ → features (8-d)
    4. Routing:    c * logits_snn + (1 - c) * logits_fused

During training, BOTH paths always run (gradient flow requires it).
During inference, the gate's value indicates how much the quantum
path contributes per sample — "easy" stages use SNN only, "hard"
stages invoke quantum features.

Input:  (batch, 6, 3000) — raw EEG signal
Output: (batch, num_classes) — classification logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..snn_1d.snn_classifier import SNN1D
from ..quantum.quantum_1d import Quantum1DEEG


class ConditionalRoutingFusion(nn.Module):
    """
    Conditional routing: SNN fast path + Quantum slow path with learned gating.

    The model learns a confidence score c ∈ [0, 1] from SNN features.
    High confidence → rely on fast SNN path.
    Low confidence  → blend in quantum-enhanced features.

    Gating modes:
        - 'soft':     linear blend: c * snn + (1-c) * fused
        - 'adaptive': c² weighting (emphasizes high-confidence fast path)
        - 'hard':     binary routing (c > threshold → SNN only)

    Args:
        num_classes: Number of output classes (default 5)
        gate_type: Gating mode ('soft', 'adaptive', 'hard')
        confidence_threshold: Threshold for hard gating (default 0.7)
        snn_fusion_dim: SNN feature dimension (default 128)
        n_qubits: Number of qubits in quantum circuit (default 8)
        quantum_rotation: Rotation type for quantum circuit (default 'RYZ')
        quantum_entanglement: Entanglement type (default 'ring')
    """

    def __init__(
        self,
        num_classes: int = 5,
        gate_type: str = 'adaptive',
        confidence_threshold: float = 0.7,
        snn_fusion_dim: int = 128,
        n_qubits: int = 8,
        quantum_rotation: str = 'RYZ',
        quantum_entanglement: str = 'ring',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gate_type = gate_type
        self.confidence_threshold = confidence_threshold
        self.snn_fusion_dim = snn_fusion_dim
        self.n_qubits = n_qubits

        # ── Fast Path: SNN-1D with Attention ──
        self.snn_branch = SNN1D(
            in_channels=6,
            num_classes=num_classes,
            use_attention=True,
            fusion_dim=snn_fusion_dim,
        )

        # ── Slow Path: Quantum-1D ──
        self.quantum_branch = Quantum1DEEG(
            in_channels=6,
            num_classes=num_classes,
            n_qubits=n_qubits,
            n_layers=3,
            rotation=quantum_rotation,
            entanglement=quantum_entanglement,
        )

        # ── Confidence Estimator ──
        # Small MLP on SNN features → scalar confidence c ∈ [0, 1]
        self.confidence_net = nn.Sequential(
            nn.Linear(snn_fusion_dim, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # ── SNN-only classifier (fast path predictions) ──
        self.classifier_snn = nn.Sequential(
            nn.Linear(snn_fusion_dim, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

        # ── Fused classifier (SNN + Quantum features → predictions) ──
        # Quantum features (n_qubits) are expanded and concatenated with SNN features
        self.quantum_expansion = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ELU(),
            nn.Linear(64, snn_fusion_dim),
        )

        self.classifier_fusion = nn.Sequential(
            nn.Linear(snn_fusion_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        # ── Gate monitoring (stored during forward for logging) ──
        self._gate_info: Dict[str, torch.Tensor] = {}

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier init for fusion layers (branch models have their own init)."""
        for module in [self.confidence_net, self.classifier_snn,
                       self.quantum_expansion, self.classifier_fusion]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _extract_quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from quantum branch (before its classifier)."""
        # Handle channel mismatch
        if x.shape[1] < 6:
            pad = torch.zeros(
                x.shape[0], 6 - x.shape[1], x.shape[2],
                device=x.device, dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)

        compressed = self.quantum_branch.compressor(x)       # (batch, 64)
        angles = self.quantum_branch.encoder(compressed)      # (batch, n_qubits)
        quantum_state = self.quantum_branch.circuit(angles)   # (batch, 2^n_qubits)
        quantum_features = self.quantum_branch.measurement(quantum_state)  # (batch, n_qubits)
        return quantum_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditional routing.

        Args:
            x: (batch, 6, 3000) — raw EEG signal

        Returns:
            (batch, num_classes) logits
        """
        # ── 1. Fast Path: SNN features ──
        snn_features = self.snn_branch.extract_features(x)   # (batch, 128)

        # ── 2. Confidence estimation ──
        confidence = self.confidence_net(snn_features)        # (batch, 1)

        # ── 3. SNN-only predictions ──
        logits_snn = self.classifier_snn(snn_features)        # (batch, num_classes)

        # ── 4. Slow Path: Quantum features (always computed during training) ──
        quantum_features = self._extract_quantum_features(x)  # (batch, n_qubits)
        quantum_expanded = self.quantum_expansion(quantum_features)  # (batch, 128)

        # ── 5. Fused predictions ──
        fused_features = torch.cat([snn_features, quantum_expanded], dim=1)  # (batch, 256)
        logits_fused = self.classifier_fusion(fused_features)  # (batch, num_classes)

        # ── 6. Gated output ──
        if self.gate_type == 'adaptive':
            # Squared confidence emphasizes high-confidence fast path
            weight = confidence ** 2
            logits = weight * logits_snn + (1 - weight) * logits_fused

        elif self.gate_type == 'soft':
            # Linear blend
            logits = confidence * logits_snn + (1 - confidence) * logits_fused

        elif self.gate_type == 'hard':
            # Binary: SNN-only if confident, fusion if uncertain
            use_fusion = (confidence < self.confidence_threshold).float()
            logits = (1 - use_fusion) * logits_snn + use_fusion * logits_fused

        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")

        # Store gate info for monitoring
        self._gate_info = {
            'confidence_mean': confidence.mean().item(),
            'confidence_std': confidence.std().item(),
            'quantum_usage': (confidence < self.confidence_threshold).float().mean().item(),
        }

        return logits

    @property
    def gate_info(self) -> Dict[str, float]:
        """Gate statistics from last forward pass (for W&B logging)."""
        return self._gate_info

    @property
    def reg_loss(self) -> torch.Tensor:
        """Spike regularization from SNN branch."""
        return self.snn_branch.reg_loss

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────── Factory ────────────────────────

def create_conditional_routing(
    num_classes: int = 5,
    gate_type: str = 'adaptive',
    quantum_rotation: str = 'RYZ',
    quantum_entanglement: str = 'ring',
) -> ConditionalRoutingFusion:
    """Create a Conditional Routing Fusion model."""
    return ConditionalRoutingFusion(
        num_classes=num_classes,
        gate_type=gate_type,
        quantum_rotation=quantum_rotation,
        quantum_entanglement=quantum_entanglement,
    )
