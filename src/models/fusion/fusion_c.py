"""
Fusion C: Multi-Modal — SNN (1D Raw) + Swin (2D Scalogram).

The paper-worthy architecture. Each modality processes the data
representation it's naturally best at:
  - SNN-1D-Attention: Raw EEG signal (6, 3000) — temporal spike dynamics
  - Swin-Tiny: CWT scalogram (3, 224, 224) — time-frequency patterns

Gated fusion combines both with a learned gate, allowing the model
to dynamically balance temporal and spectral information.

Input:  raw_signal (batch, 6, 3000) + scalogram (batch, 3, 224, 224)
Output: (batch, 5) — sleep stage logits
"""

import torch
import torch.nn as nn
import timm

from .gated_fusion import GatedFusionModule, ClassificationHead
from ..snn_1d.snn_classifier import SNN1D


class FusionC(nn.Module):
    """
    Multi-Modal Fusion: SNN (raw 1D) + Swin (2D scalogram).
    
    Architecture:
        Raw EEG (6, 3000) → SNN-1D-Attention → 128-d features
        Scalogram (3, 224, 224) → Swin-Tiny → 768-d features
        → GatedFusion(primary=256, secondary=256) → 256-d
        → Classifier → 5 classes
    
    The SNN captures temporal spike dynamics directly from the raw signal.
    The Swin captures time-frequency patterns from the scalogram.
    Together they provide complementary views of the same EEG epoch.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        fusion_dim: int = 256,
    ):
        super().__init__()
        
        # ── SNN branch (raw 1D signal) ──
        self.snn = SNN1D(
            in_channels=6,
            num_classes=num_classes,
            use_attention=True,     # Use the attention variant
            fusion_dim=128,
        )
        snn_feature_dim = 128
        
        # ── Swin branch (2D scalogram) ──
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0,
        )
        swin_feature_dim = self.swin.num_features  # 768
        
        # ── Project both to fusion_dim ──
        self.snn_proj = nn.Sequential(
            nn.Linear(snn_feature_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ELU(),
        )
        self.swin_proj = nn.Sequential(
            nn.Linear(swin_feature_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ELU(),
        )
        
        # ── Gated Fusion ──
        self.fusion = GatedFusionModule(
            primary_dim=fusion_dim,
            secondary_dim=fusion_dim,
            gate_hidden=128,
        )
        
        # ── Classifier ──
        self.classifier = ClassificationHead(
            input_dim=fusion_dim,
            hidden_dim=128,
            num_classes=num_classes,
        )
    
    def _extract_snn_features(self, raw_signal: torch.Tensor) -> torch.Tensor:
        """
        Run SNN forward but extract features BEFORE the classifier.
        
        Args:
            raw_signal: (batch, 6, 3000)
        Returns:
            (batch, 128) — SNN features
        """
        snn = self.snn
        
        # Handle channel mismatch
        x = raw_signal
        if x.shape[1] < 6:
            pad = torch.zeros(
                x.shape[0], 6 - x.shape[1], x.shape[2],
                device=x.device, dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)
        
        # Three-stage pyramid
        s1, reg1 = snn.stage1(x)
        s2, reg2 = snn.stage2(s1)
        s3, reg3 = snn.stage3(s2)
        
        # Temporal fusion (features, not logits)
        features = snn.fusion(s1, s2, s3)  # (batch, 128)
        
        # Store reg loss
        self._snn_reg_loss = reg1 + reg2 + reg3
        
        return features
    
    def forward(
        self,
        raw_signal: torch.Tensor,
        scalogram: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            raw_signal: (batch, 6, 3000) — raw EEG
            scalogram: (batch, 3, 224, 224) — CWT scalogram
        Returns:
            (batch, num_classes) logits
        """
        # Extract features from both modalities
        snn_features = self._extract_snn_features(raw_signal)  # (batch, 128)
        swin_features = self.swin(scalogram)                    # (batch, 768)
        
        # Project to common dimension
        snn_proj = self.snn_proj(snn_features)                  # (batch, 256)
        swin_proj = self.swin_proj(swin_features)               # (batch, 256)
        
        # Gated fusion
        fused, self._gate = self.fusion(swin_proj, snn_proj)    # (batch, 256)
        
        # Classify
        logits = self.classifier(fused)
        return logits
    
    @property
    def snn_reg_loss(self) -> torch.Tensor:
        """SNN spike regularization loss."""
        return getattr(self, '_snn_reg_loss', torch.tensor(0.0))
    
    @property
    def gate_value(self) -> torch.Tensor:
        """Fusion gate value (0 = Swin only, 1 = SNN only)."""
        return getattr(self, '_gate', torch.tensor(0.5))
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_fusion_c(num_classes: int = 5, pretrained: bool = True) -> FusionC:
    """Create Fusion-C: Multi-Modal SNN(1D) + Swin(2D)."""
    return FusionC(num_classes=num_classes, pretrained=pretrained)
