"""
Swin Transformer Wrapper for ECG Classification.

This module provides a wrapper around the timm Swin Transformer implementation,
adapted for ECG classification using scalogram images.

Reference:
- Swin Transformer: https://arxiv.org/abs/2103.14030
- timm: https://github.com/huggingface/pytorch-image-models
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import timm


class SwinECGClassifier(nn.Module):
    """
    Swin Transformer for ECG Scalogram Classification.
    
    Wraps the timm Swin Transformer with customizations for ECG classification:
    - Supports variable number of input channels
    - Custom classification head
    - Optional feature extraction mode
    
    Args:
        num_classes: Number of output classes. Default: 5.
        model_name: Swin variant from timm. Default: 'swin_tiny_patch4_window7_224'.
        pretrained: Use ImageNet pretrained weights. Default: True.
        in_channels: Number of input channels. Default: 3.
        drop_rate: Dropout rate for classifier. Default: 0.0.
        drop_path_rate: Stochastic depth rate. Default: 0.1.
        freeze_backbone: Freeze backbone for linear probing. Default: False.
    
    Example:
        >>> model = SwinECGClassifier(num_classes=5, pretrained=True)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> output = model(x)  # (4, 5)
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        model_name: str = 'swin_tiny_patch4_window7_224',
        pretrained: bool = True,
        in_channels: int = 3,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.in_channels = in_channels
        
        # Create base model from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            in_chans=in_channels,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        # Get feature dimension from backbone
        self.feature_dim = self._get_feature_dim()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(drop_rate),
            nn.Linear(self.feature_dim, num_classes),
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _get_feature_dim(self) -> int:
        """Determine feature dimension from backbone."""
        # Different Swin variants have different embedding dimensions
        if hasattr(self.backbone, 'num_features'):
            return self.backbone.num_features
        elif hasattr(self.backbone, 'embed_dim'):
            # Swin: embed_dim * 8 (after 4 stages with 2x multiplier each)
            return self.backbone.embed_dim * 8
        else:
            # Fallback: run a dummy forward pass
            with torch.no_grad():
                dummy = torch.zeros(1, self.in_channels, 224, 224)
                features = self.backbone(dummy)
                return features.shape[-1]
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Froze {sum(1 for _ in self.backbone.parameters())} backbone parameters")
    
    def unfreeze_backbone(self, unfreeze_last_n: Optional[int] = None):
        """
        Unfreeze backbone parameters.
        
        Args:
            unfreeze_last_n: If provided, only unfreeze last N layers.
                           If None, unfreeze all.
        """
        if unfreeze_last_n is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last N transformer blocks
            layers = list(self.backbone.layers)
            for layer in layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Feature tensor of shape (B, feature_dim).
        """
        return self.backbone(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
               Expected size: 224x224 for default model.
            
        Returns:
            Output logits of shape (B, num_classes).
        """
        # Extract features
        features = self.backbone(x)  # (B, feature_dim)
        
        # Classification
        logits = self.classifier(features)  # (B, num_classes)
        
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention maps from all transformer blocks (for visualization).
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Dictionary mapping layer names to attention tensors.
        """
        attention_maps = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                # For Swin, attention is computed within WindowAttention
                if hasattr(module, 'attn'):
                    attention_maps[name] = module.attn.detach()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in self.backbone.named_modules():
            if 'attn' in name.lower():
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        with torch.no_grad():
            _ = self.backbone(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps


class SwinECGEnsemble(nn.Module):
    """
    Ensemble of Swin Transformers with different configurations.
    
    Combines multiple Swin variants for improved robustness.
    
    Args:
        num_classes: Number of output classes. Default: 5.
        model_configs: List of model configuration dicts.
        aggregation: How to combine predictions ('mean', 'vote', 'learned').
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        model_configs: Optional[list] = None,
        aggregation: str = 'mean',
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.aggregation = aggregation
        
        if model_configs is None:
            model_configs = [
                {'model_name': 'swin_tiny_patch4_window7_224'},
                {'model_name': 'swin_small_patch4_window7_224'},
            ]
        
        # Create ensemble members
        self.models = nn.ModuleList([
            SwinECGClassifier(num_classes=num_classes, **config)
            for config in model_configs
        ])
        
        if aggregation == 'learned':
            # Learnable combination weights
            self.combination_weights = nn.Parameter(
                torch.ones(len(self.models)) / len(self.models)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ensemble aggregation.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Aggregated output logits of shape (B, num_classes).
        """
        # Get predictions from all models
        predictions = [model(x) for model in self.models]
        predictions = torch.stack(predictions, dim=0)  # (N_models, B, classes)
        
        if self.aggregation == 'mean':
            return predictions.mean(dim=0)
        elif self.aggregation == 'vote':
            # Soft voting (average probabilities)
            probs = torch.softmax(predictions, dim=-1)
            return probs.mean(dim=0)
        elif self.aggregation == 'learned':
            weights = torch.softmax(self.combination_weights, dim=0)
            weights = weights.view(-1, 1, 1)
            return (predictions * weights).sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


class SwinWithAuxiliaryHead(nn.Module):
    """
    Swin Transformer with auxiliary classification heads for multi-task learning.
    
    Includes:
    - Main classification head (diagnostic class)
    - Auxiliary heads for related tasks (e.g., age prediction, sex classification)
    
    Args:
        num_classes: Number of main classification classes. Default: 5.
        aux_tasks: Dict mapping task names to number of classes/outputs.
        **kwargs: Arguments passed to SwinECGClassifier.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        aux_tasks: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        super().__init__()
        
        if aux_tasks is None:
            aux_tasks = {}
        
        # Main model
        self.swin = SwinECGClassifier(num_classes=num_classes, **kwargs)
        
        # Auxiliary heads
        self.aux_heads = nn.ModuleDict({
            task: nn.Linear(self.swin.feature_dim, n_out)
            for task, n_out in aux_tasks.items()
        })
    
    def forward(
        self, x: torch.Tensor, return_aux: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with optional auxiliary outputs.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            return_aux: Whether to return auxiliary predictions.
            
        Returns:
            Tuple of (main_logits, aux_outputs_dict).
        """
        # Get features
        features = self.swin.get_features(x)
        
        # Main classification
        main_logits = self.swin.classifier(features)
        
        if return_aux and self.aux_heads:
            aux_outputs = {
                task: head(features)
                for task, head in self.aux_heads.items()
            }
            return main_logits, aux_outputs
        
        return main_logits, None


def create_swin_classifier(
    model_name: str = 'swin_tiny_patch4_window7_224',
    num_classes: int = 5,
    pretrained: bool = True,
    **kwargs,
) -> SwinECGClassifier:
    """
    Factory function to create Swin Transformer ECG classifiers.
    
    Args:
        model_name: Swin variant name. Options:
            - 'swin_tiny_patch4_window7_224' (default, 28M params)
            - 'swin_small_patch4_window7_224' (50M params)
            - 'swin_base_patch4_window7_224' (88M params)
            - 'swin_large_patch4_window7_224' (197M params)
        num_classes: Number of output classes.
        pretrained: Use ImageNet pretrained weights.
        **kwargs: Additional arguments.
        
    Returns:
        SwinECGClassifier model.
    """
    return SwinECGClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        **kwargs,
    )


# Available Swin models in timm
AVAILABLE_MODELS = [
    'swin_tiny_patch4_window7_224',
    'swin_small_patch4_window7_224',
    'swin_base_patch4_window7_224',
    'swin_base_patch4_window12_384',
    'swin_large_patch4_window7_224',
    'swin_large_patch4_window12_384',
    # V2 variants
    'swinv2_tiny_window8_256',
    'swinv2_small_window8_256',
    'swinv2_base_window8_256',
]


def list_available_models() -> list:
    """Return list of available Swin model names."""
    return AVAILABLE_MODELS.copy()
