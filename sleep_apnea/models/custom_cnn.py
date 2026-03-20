"""
Custom CNN Baseline for Sleep Apnea Classification.

Simple but effective CNN architecture for baseline comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CustomCNN(nn.Module):
    """
    Custom CNN for sleep apnea severity classification.
    
    Architecture:
        Input: (batch, in_channels, time) or (batch, in_channels, H, W)
        4x Conv blocks with BatchNorm + ReLU + MaxPool
        Global Average Pooling
        FC head: Linear -> ReLU -> Dropout -> Linear
    
    Args:
        in_channels: Number of input channels (EEG channels or scalogram channels)
        num_classes: Number of output classes (default: 4)
        base_filters: Base number of filters (doubles each block)
        num_blocks: Number of conv blocks
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 4,
        base_filters: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_ch = in_channels
        for i in range(num_blocks):
            out_ch = base_filters * (2 ** i)
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                )
            )
            in_ch = out_ch
        
        # Compute feature dimension after pooling
        # Assuming input length ~3000, after 4 pooling: 3000 -> 1500 -> 750 -> 375 -> 187
        self.feature_dim = base_filters * (2 ** (num_blocks - 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, time) or (batch, channels, H, W)
        
        Returns:
            Logits (batch, num_classes)
        """
        # Handle 2D input (scalograms)
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C * H, W)  # Flatten spatial dims into channels
        
        # Conv blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Global average pooling
        x = x.mean(dim=-1)  # (batch, feature_dim)
        
        # Classification
        return self.classifier(x)
    
    def get_num_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


class CustomCNN2D(nn.Module):
    """
    Custom CNN for 2D scalogram input.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        base_filters: int = 32,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        # Conv blocks (2D)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, 3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(base_filters * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(1)
        return self.classifier(x)


def create_cnn_baseline(
    in_channels: int = 6,
    num_classes: int = 4,
    pretrained: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Factory function for Custom CNN baseline.
    
    Args:
        in_channels: Input channels
        num_classes: Output classes
        pretrained: Not used (no pretrained weights available)
    
    Returns:
        CustomCNN model
    """
    return CustomCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        **kwargs,
    )


if __name__ == '__main__':
    # Test model
    model = create_cnn_baseline(in_channels=6, num_classes=4)
    x = torch.randn(4, 6, 3000)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Parameters: {model.get_num_parameters():,}")
