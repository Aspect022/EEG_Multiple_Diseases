"""Transformer and CNN-based models for EEG Sleep Staging."""

from .swin import (
    SwinECGClassifier,
    SwinECGEnsemble,
    SwinWithAuxiliaryHead,
    create_swin_classifier,
    list_available_models,
    AVAILABLE_MODELS,
)

from .vit import ViTSmall, create_vit_classifier
from .deit import DeiTSmall, create_deit_classifier
from .efficientnet import EfficientNetB0, create_efficientnet_classifier
from .convnext import ConvNeXtTiny, create_convnext_classifier

__all__ = [
    # Swin
    'SwinECGClassifier',
    'SwinECGEnsemble',
    'SwinWithAuxiliaryHead',
    'create_swin_classifier',
    'list_available_models',
    'AVAILABLE_MODELS',
    # ViT
    'ViTSmall',
    'create_vit_classifier',
    # DeiT
    'DeiTSmall',
    'create_deit_classifier',
    # EfficientNet
    'EfficientNetB0',
    'create_efficientnet_classifier',
    # ConvNeXt
    'ConvNeXtTiny',
    'create_convnext_classifier',
]
