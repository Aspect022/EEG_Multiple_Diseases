"""
Vision Transformer (ViT-Small) for EEG Sleep Staging.

Uses timm's pretrained ViT-Small with fine-tuned classification head.
Input: 224×224 scalogram images (3-channel).
"""

import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class ViTSmall(nn.Module):
    """
    ViT-Small classifier for 224×224 scalogram images.

    Uses timm's vit_small_patch16_224 with ImageNet pretrained weights.
    The classification head is replaced with a custom 5-class head.

    Args:
        num_classes: Number of output classes (default: 5)
        pretrained: Use ImageNet pretrained weights (default: True)
        drop_rate: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        drop_rate: float = 0.1,
    ):
        super().__init__()

        if not HAS_TIMM:
            raise ImportError("timm is required: pip install timm")

        self.model = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 224, 224) scalogram images
        Returns:
            (batch, num_classes) logits
        """
        return self.model(x)


def create_vit_classifier(
    num_classes: int = 5,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Factory function for ViT-Small."""
    return ViTSmall(
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs,
    )
