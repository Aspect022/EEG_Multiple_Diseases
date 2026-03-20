"""
Fusion A: Swin + ConvNeXt — Classical Dual-Backbone Fusion.

Combines Swin Transformer's local window attention with
ConvNeXt's hierarchical CNN features via gated fusion.

Both process 2D scalograms. Backbones are fine-tuned end-to-end.

Input:  (batch, 3, 224, 224) — CWT scalogram
Output: (batch, 5)           — sleep stage logits
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple

from .gated_fusion import GatedFusionModule, ClassificationHead


class FusionA(nn.Module):
    """
    Swin-Tiny + ConvNeXt-Tiny with gated fusion.
    
    Architecture:
        Scalogram → Swin-Tiny → 768-d features
        Scalogram → ConvNeXt-Tiny → 768-d features
        → GatedFusion → 768-d → Classifier → 5 classes
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        fusion_dim: int = 768,
    ):
        super().__init__()
        
        # Swin-Tiny backbone
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier, get features
        )
        swin_dim = self.swin.num_features  # 768
        
        # ConvNeXt-Tiny backbone
        self.convnext = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            num_classes=0,
        )
        convnext_dim = self.convnext.num_features  # 768
        
        # Gated fusion
        self.fusion = GatedFusionModule(
            primary_dim=swin_dim,
            secondary_dim=convnext_dim,
            gate_hidden=256,
        )
        
        # Classifier
        self.classifier = ClassificationHead(
            input_dim=swin_dim,
            hidden_dim=256,
            num_classes=num_classes,
        )
    
    def forward(self, x: torch.Tensor = None, scalogram: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 224, 224) — scalogram (for API compatibility)
            scalogram: (batch, 3, 224, 224) — scalogram (for API compatibility)
        Returns:
            (batch, num_classes) logits
        """
        # Handle both x and scalogram for API compatibility
        if x is None and scalogram is not None:
            x = scalogram
        elif x is None:
            raise ValueError("Either x or scalogram must be provided")
        
        # Extract features from both backbones
        swin_features = self.swin(x)        # (batch, 768)
        convnext_features = self.convnext(x)  # (batch, 768)

        # Gated fusion
        fused, self._gate = self.fusion(swin_features, convnext_features)

        # Classify
        logits = self.classifier(fused)
        return logits
    
    @property
    def gate_value(self) -> torch.Tensor:
        return getattr(self, '_gate', torch.tensor(0.5))
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_fusion_a(num_classes: int = 5, pretrained: bool = True) -> FusionA:
    """Create Fusion-A: Swin + ConvNeXt."""
    return FusionA(num_classes=num_classes, pretrained=pretrained)
