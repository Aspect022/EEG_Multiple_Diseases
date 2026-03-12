"""
Data-efficient Image Transformer (DeiT-Small) for EEG Sleep Staging.

DeiT was designed to work well WITHOUT large-scale pretraining data.
Uses knowledge distillation from a CNN teacher during training.

Input: 224×224 scalogram images (3-channel).
"""

import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class DeiTSmall(nn.Module):
    """
    DeiT-Small classifier for 224×224 scalogram images.

    Uses timm's deit_small_patch16_224 with ImageNet pretrained weights.
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
            'deit_small_patch16_224',
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


def create_deit_classifier(
    num_classes: int = 5,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Factory function for DeiT-Small."""
    return DeiTSmall(
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs,
    )
