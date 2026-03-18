"""
Fusion B: Swin + ConvNeXt + DeiT + Quantum — 4-Way Hybrid Fusion.

Combines four complementary architectures:
  - Swin: Local window attention (periodic/rhythmic EEG patterns)
  - ConvNeXt: Hierarchical CNN (spatial scalogram structure)
  - DeiT: Global self-attention (long-range dependencies)
  - Quantum: Novel quantum feature space representation

All process 2D scalograms. Uses MultiStreamFusion with learned attention.

Input:  (batch, 3, 224, 224) — CWT scalogram
Output: (batch, 5)           — sleep stage logits
"""

import torch
import torch.nn as nn
import timm

from .gated_fusion import MultiStreamFusion, ClassificationHead
from ..quantum.hybrid_cnn import create_hybrid_quantum_cnn


class FusionB(nn.Module):
    """
    4-way hybrid fusion: Swin + ConvNeXt + DeiT + Quantum-ring-RXY.
    
    Architecture:
        Scalogram → Swin → 768-d
        Scalogram → ConvNeXt → 768-d
        Scalogram → DeiT → 384-d
        Scalogram → Q-ring-RXY → 5-d (logits used as features)
        → MultiStreamFusion → 768-d → Classifier → 5 classes
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        fusion_dim: int = 768,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Swin-Tiny backbone
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0,
        )
        swin_dim = self.swin.num_features  # 768
        
        # ConvNeXt-Tiny backbone
        self.convnext = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            num_classes=0,
        )
        convnext_dim = self.convnext.num_features  # 768
        
        # DeiT-Small backbone
        self.deit = timm.create_model(
            'deit_small_patch16_224',
            pretrained=pretrained,
            num_classes=0,
        )
        deit_dim = self.deit.num_features  # 384
        
        # Quantum backbone (ring-RXY)
        # Use the 2D hybrid quantum CNN since this is scalogram-based
        self.quantum = create_hybrid_quantum_cnn(
            num_classes=num_classes,
            entanglement_type='ring',
            rotation_type='RXY',
            n_qubits=8,
            n_qlayers=3,
        )
        # We'll extract the intermediate features, not final logits
        quantum_feature_dim = 8  # n_qubits measurement output
        
        # Multi-stream fusion
        self.fusion = MultiStreamFusion(
            stream_dims=[swin_dim, convnext_dim, deit_dim, quantum_feature_dim],
            fusion_dim=fusion_dim,
        )
        
        # Classifier
        self.classifier = ClassificationHead(
            input_dim=fusion_dim,
            hidden_dim=256,
            num_classes=num_classes,
        )
    
    def _extract_quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate quantum features (before classifier)."""
        # Run through quantum model's encoder → pre_quantum → circuit → measurement
        model = self.quantum
        batch_size = x.shape[0]
        features = model.encoder(x)                          # (B, 128, 2, 2)
        flat = features.view(batch_size, -1)                  # (B, 512)
        angles = model.pre_quantum(flat)                      # (B, n_qubits)
        angles = torch.tanh(angles) * 3.14159                 # [-π, π]
        
        quantum_state = model.circuit(angles)                  # (B, 2^n_qubits)
        quantum_features = model.measurement(quantum_state)    # (B, n_qubits)
        return quantum_features.float()  # (batch, 8)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 224, 224) — scalogram
        Returns:
            (batch, num_classes) logits
        """
        # Extract features from all 4 backbones
        swin_feat = self.swin(x)           # (batch, 768)
        convnext_feat = self.convnext(x)   # (batch, 768)
        deit_feat = self.deit(x)           # (batch, 384)
        quantum_feat = self._extract_quantum_features(x)  # (batch, 8)
        
        # 4-way fusion
        fused = self.fusion([swin_feat, convnext_feat, deit_feat, quantum_feat])
        
        # Classify
        logits = self.classifier(fused)
        return logits
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_fusion_b(num_classes: int = 5, pretrained: bool = True) -> FusionB:
    """Create Fusion-B: Swin + ConvNeXt + DeiT + Quantum 4-way."""
    return FusionB(num_classes=num_classes, pretrained=pretrained)
