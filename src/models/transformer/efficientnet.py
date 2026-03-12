"""
EfficientNet-B0 for EEG Sleep Staging.

Lightweight CNN baseline with compound scaling.
Strong performer with very few parameters (~5.3M).

Input: 224×224 scalogram images (3-channel).
"""

import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 classifier for 224×224 scalogram images.

    Uses timm's efficientnet_b0 with ImageNet pretrained weights.

    Args:
        num_classes: Number of output classes (default: 5)
        pretrained: Use ImageNet pretrained weights (default: True)
        drop_rate: Dropout rate (default: 0.2)
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        drop_rate: float = 0.2,
    ):
        super().__init__()

        if not HAS_TIMM:
            raise ImportError("timm is required: pip install timm")

        self.model = timm.create_model(
            'efficientnet_b0',
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


def create_efficientnet_classifier(
    num_classes: int = 5,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Factory function for EfficientNet-B0."""
    return EfficientNetB0(
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs,
    )
