"""
Hybrid ViT + BiLSTM for Sleep Apnea Classification.

Main contribution model: Combines spatial feature extraction (ViT) 
with temporal sequence modeling (BiLSTM) for full-night analysis.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class ViTBiLSTMHybrid(nn.Module):
    """
    Hybrid Vision Transformer + Bidirectional LSTM for sleep apnea classification.
    
    Architecture:
        1. Per-epoch encoding: ViT extracts features from each epoch's scalogram
        2. Temporal modeling: BiLSTM processes sequence of epoch features
        3. Classification: Head processes final hidden state
    
    Input: (batch, num_epochs, channels, height, width)
           Example: (4, 30, 3, 224, 224) for 30 epochs per night
    
    Output: (batch, num_classes) - Recording-level severity prediction
    
    Args:
        vit_name: ViT variant from timm
        vit_pretrained: Use ImageNet pretrained ViT
        lstm_hidden: LSTM hidden size
        lstm_layers: Number of LSTM layers
        lstm_dropout: LSTM dropout
        num_classes: Number of output classes
        num_epochs_sequence: Number of epochs per recording
    """
    
    def __init__(
        self,
        vit_name: str = 'vit_small_patch16_224',
        vit_pretrained: bool = True,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        num_classes: int = 4,
        num_epochs_sequence: int = 30,
    ):
        super().__init__()
        
        if not HAS_TIMM:
            raise ImportError("timm is required: pip install timm")
        
        self.vit_name = vit_name
        self.num_classes = num_classes
        self.num_epochs_sequence = num_epochs_sequence
        
        # ViT encoder (per-epoch feature extraction)
        self.vit = timm.create_model(
            vit_name,
            pretrained=vit_pretrained,
            num_classes=0,  # Remove classification head
        )
        vit_dim = self.vit.num_features
        
        # Temporal modeling with BiLSTM
        self.lstm = nn.LSTM(
            input_size=vit_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),  # *2 for bidirectional
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(lstm_dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, num_epochs, channels, height, width)
               Example: (4, 30, 3, 224, 224)
        
        Returns:
            Logits (batch, num_classes)
        """
        B, T, C, H, W = x.shape
        
        # Reshape for ViT: (B*T, C, H, W)
        x = x.view(B * T, C, H, W)
        
        # ViT encoding (per epoch)
        vit_features = self.vit(x)  # (B*T, vit_dim)
        
        # Reshape back to sequence: (B, T, vit_dim)
        vit_features = vit_features.view(B, T, -1)
        
        # LSTM temporal modeling
        lstm_out, (h_n, c_n) = self.lstm(vit_features)
        # lstm_out: (B, T, 2*hidden)
        # h_n: (2*layers, B, hidden)
        
        # Concatenate final forward and backward hidden states
        # h_n shape: (num_layers * 2, B, hidden)
        h_forward = h_n[-2]  # Last layer forward
        h_backward = h_n[-1]  # Last layer backward
        final_hidden = torch.cat([h_forward, h_backward], dim=1)  # (B, 2*hidden)
        
        # Classification
        return self.classifier(final_hidden)
    
    def get_num_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_vit_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract ViT features without LSTM/classifier.
        
        Args:
            x: (batch, num_epochs, channels, height, width)
        
        Returns:
            Features: (batch, num_epochs, vit_dim)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        vit_features = self.vit(x)
        return vit_features.view(B, T, -1)


class ViTBiLSTMWithAttention(nn.Module):
    """
    ViT + BiLSTM with attention pooling over epochs.
    
    Instead of using only final hidden state, applies attention
    to weight importance of different epochs.
    """
    
    def __init__(
        self,
        vit_name: str = 'vit_small_patch16_224',
        vit_pretrained: bool = True,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        num_classes: int = 4,
        attention_dim: int = 128,
    ):
        super().__init__()
        
        if not HAS_TIMM:
            raise ImportError("timm is required: pip install timm")
        
        # ViT encoder
        self.vit = timm.create_model(
            vit_name,
            pretrained=vit_pretrained,
            num_classes=0,
        )
        vit_dim = self.vit.num_features
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=vit_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        
        # ViT encoding
        x = x.view(B * T, C, H, W)
        vit_features = self.vit(x)
        vit_features = vit_features.view(B, T, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(vit_features)  # (B, T, 2*hidden)
        
        # Attention weights
        attn_weights = self.attention(lstm_out)  # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, 2*hidden)
        
        # Classification
        return self.classifier(context)


def create_vit_bilstm(
    vit_name: str = 'vit_small_patch16_224',
    vit_pretrained: bool = True,
    lstm_hidden: int = 256,
    lstm_layers: int = 2,
    num_classes: int = 4,
    use_attention: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Factory function for ViT+BiLSTM models.
    
    Args:
        vit_name: ViT variant
        vit_pretrained: Use pretrained weights
        lstm_hidden: LSTM hidden size
        lstm_layers: Number of LSTM layers
        num_classes: Output classes
        use_attention: Use attention pooling
    
    Returns:
        ViT+BiLSTM model
    """
    if use_attention:
        return ViTBiLSTMWithAttention(
            vit_name=vit_name,
            vit_pretrained=vit_pretrained,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            num_classes=num_classes,
            **kwargs,
        )
    else:
        return ViTBiLSTMHybrid(
            vit_name=vit_name,
            vit_pretrained=vit_pretrained,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            num_classes=num_classes,
            **kwargs,
        )


if __name__ == '__main__':
    # Test model
    model = create_vit_bilstm(num_classes=4)
    
    # Simulate batch of 4 recordings, 30 epochs each
    x = torch.randn(4, 30, 3, 224, 224)
    y = model(x)
    
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Parameters: {model.get_num_parameters():,}")
