"""
ResNet18 Transfer Learning for Sleep Apnea Classification.

Uses ImageNet-pretrained ResNet18 with modified first layer for EEG input.
"""

import torch
import torch.nn as nn
from typing import Optional, List

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class ResNet18Transfer(nn.Module):
    """
    ResNet18 with transfer learning for sleep apnea classification.
    
    Strategy:
        1. Start with ImageNet pretrained weights
        2. Modify first conv layer for EEG channels (6 vs 3)
        3. Progressive unfreezing:
           - Phase 1: Freeze backbone, train head
           - Phase 2: Unfreeze layer4, fine-tune
           - Phase 3: Full fine-tuning (optional)
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        freeze_layers: List of layer indices to freeze initially
    """
    
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_layers: List[int] = [0, 1],  # Freeze conv1, layer1, layer2
    ):
        super().__init__()
        
        if not HAS_TIMM:
            raise ImportError("timm is required: pip install timm")
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Create ResNet18 backbone (without classification head)
        self.backbone = timm.create_model(
            'resnet18',
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,  # Will modify for our channels
        )
        
        # Modify first convolution for our input channels
        if in_channels != 3:
            old_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                bias=old_conv1.bias,
            )
            # Initialize with pretrained weights (repeat or zero-pad)
            with torch.no_grad():
                if in_channels > 3:
                    # Repeat pretrained weights
                    old_weight = old_conv1.weight  # (64, 3, 7, 7)
                    new_weight = old_weight.repeat(1, in_channels // 3 + 1, 1, 1)[:, :in_channels, :, :]
                else:
                    # Take first channels
                    new_weight = old_weight[:, :in_channels, :, :]
                self.backbone.conv1.weight.copy_(new_weight)
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        
        # Freeze layers initially
        self._freeze_layers(freeze_layers)
    
    def _freeze_layers(self, layer_indices: List[int]):
        """Freeze specified layers."""
        layers = [self.backbone.conv1, self.backbone.layer1, 
                  self.backbone.layer2, self.backbone.layer3, 
                  self.backbone.layer4]
        
        for i in layer_indices:
            if i < len(layers):
                for param in layers[i].parameters():
                    param.requires_grad = False
        
        print(f"  [ResNet18] Froze layers: {layer_indices}")
    
    def unfreeze_all(self):
        """Unfreeze all backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("  [ResNet18] Unfroze all layers")
    
    def unfreeze_layer(self, layer_idx: int):
        """Unfreeze a specific layer."""
        layers = [self.backbone.conv1, self.backbone.layer1, 
                  self.backbone.layer2, self.backbone.layer3, 
                  self.backbone.layer4]
        if layer_idx < len(layers):
            for param in layers[layer_idx].parameters():
                param.requires_grad = True
            print(f"  [ResNet18] Unfroze layer {layer_idx}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, in_channels, H, W)
        
        Returns:
            Logits (batch, num_classes)
        """
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_resnet18_transfer(
    in_channels: int = 6,
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_layers: Optional[List[int]] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function for ResNet18 transfer learning.
    
    Args:
        in_channels: Input channels
        num_classes: Output classes
        pretrained: Use ImageNet weights
        freeze_layers: Layers to freeze initially
    
    Returns:
        ResNet18Transfer model
    """
    if freeze_layers is None:
        freeze_layers = [0, 1]  # Default: freeze conv1 and layer1
    
    return ResNet18Transfer(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_layers=freeze_layers,
        **kwargs,
    )


if __name__ == '__main__':
    # Test model
    model = create_resnet18_transfer(in_channels=6, num_classes=4)
    x = torch.randn(4, 6, 224, 224)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Trainable parameters: {model.get_num_parameters():,}")
