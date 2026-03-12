"""
ConvNeXt-Tiny for EEG Sleep Staging.

Modern pure-CNN architecture that matches or beats transformers.
Represents state-of-the-art CNN design (2022+).

Input: 224×224 scalogram images (3-channel).
"""

import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class ConvNeXtTiny(nn.Module):
    """
    ConvNeXt-Tiny classifier for 224×224 scalogram images.

    Uses timm's convnext_tiny with ImageNet pretrained weights.

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
            'convnext_tiny',
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


def create_convnext_classifier(
    num_classes: int = 5,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Factory function for ConvNeXt-Tiny."""
    return ConvNeXtTiny(
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs,
    )
