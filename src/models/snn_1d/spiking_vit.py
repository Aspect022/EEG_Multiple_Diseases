"""
Spiking Vision Transformer for 1D EEG signals.

Ported from ECG project's ViT1D with SpikingTransformerBlock adaptation.
Patches raw EEG signal, applies spiking transformer blocks, classifies via CLS token.

Input:  (batch, 6, 3000)  — 6 EEG channels, 30s @ 100Hz
Output: (batch, 5)        — 5 sleep stage classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from dataclasses import dataclass

from .attention import MultiScaleSpikingAttention


# ──────────────────────── Config ────────────────────────

@dataclass
class SpikingViT1DConfig:
    """Configuration for Spiking ViT 1D."""
    # Input
    in_channels: int = 6
    input_length: int = 3000
    num_classes: int = 5
    
    # Patch embedding
    patch_size: int = 30        # 30 samples per patch → 100 tokens
    embed_dim: int = 128
    
    # Transformer
    depth: int = 4              # number of transformer blocks
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    
    # Classifier
    hidden_dim: int = 64
    cls_dropout: float = 0.3


# ──────────────────────── Patch Embedding ────────────────────────

class PatchEmbedding1D(nn.Module):
    """
    Split EEG signal into non-overlapping patches and embed.
    
    (batch, 6, 3000) → (batch, 100, embed_dim)
    """
    
    def __init__(
        self,
        in_channels: int = 6,
        patch_size: int = 30,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, length)
        Returns:
            (batch, num_patches, embed_dim)
        """
        x = self.proj(x)           # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)      # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


# ──────────────────────── Spiking Transformer Block ────────────────────────

class SpikingTransformerBlock(nn.Module):
    """
    Transformer block with MHSA replaced by MultiScaleSpikingAttention.
    
    The MSSA operates at three scales:
    - Local: captures fine temporal patterns within patches
    - Regional: captures cross-patch relationships
    - Global: captures epoch-wide patterns
    """
    
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        # MSSA expects (B, C, L) and returns (B, C, L)
        self.mssa = MultiScaleSpikingAttention(
            dim=dim,
            local_window=32,
            regional_window=64,     # Adjusted for ~100 tokens
            global_pool=8,
        )
        
        self.norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, length, dim)
        Returns:
            (batch, length, dim)
        """
        # MSSA expects (B, C, L), so transpose in and out
        x_conv = x.transpose(1, 2)         # (B, dim, L)
        x_conv = self.mssa(x_conv)          # (B, dim, L)
        x = x + x_conv.transpose(1, 2)     # residual in (B, L, dim)
        
        # FFN + residual
        x = x + self.ffn(self.norm(x))
        return x


# ──────────────────────── Spiking ViT 1D ────────────────────────

class SpikingViT1D(nn.Module):
    """
    Spiking Vision Transformer for 1D EEG signals.
    
    Architecture:
        Raw EEG (6, 3000) → PatchEmbed → 100 tokens → CLS token → 
        4× SpikingTransformerBlock → CLS features → Classifier → 5 classes
    """
    
    def __init__(self, config: Optional[SpikingViT1DConfig] = None):
        super().__init__()
        if config is None:
            config = SpikingViT1DConfig()
        self.config = config
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding1D(
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
        )
        num_patches = config.input_length // config.patch_size
        
        # CLS token + Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.embed_dim),
        )
        self.pos_drop = nn.Dropout(config.dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Spiking Transformer Blocks
        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(
                dim=config.embed_dim,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
            )
            for _ in range(config.depth)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ELU(),
            nn.Dropout(config.cls_dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 6, 3000) — raw EEG signal
        Returns:
            (batch, num_classes) logits
        """
        B = x.shape[0]
        
        # Handle channel mismatch
        if x.shape[1] < self.config.in_channels:
            pad = torch.zeros(
                B, self.config.in_channels - x.shape[1], x.shape[2],
                device=x.device, dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)
        
        # Handle length mismatch
        if x.shape[2] != self.config.input_length:
            x = F.interpolate(
                x, size=self.config.input_length, mode='linear', align_corners=False,
            )
        
        # Patch embedding
        tokens = self.patch_embed(x)                    # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)        # (B, N+1, D)
        tokens = self.pos_drop(tokens + self.pos_embed)
        
        # Spiking transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens)
        
        tokens = self.norm(tokens)
        cls_features = tokens[:, 0]                     # (B, D)
        
        # Classify
        logits = self.classifier(cls_features)
        
        return logits
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────── Factory ────────────────────────

def create_spiking_vit_1d(
    num_classes: int = 5,
    in_channels: int = 6,
) -> SpikingViT1D:
    """Create Spiking-ViT-1D model."""
    config = SpikingViT1DConfig(
        in_channels=in_channels,
        num_classes=num_classes,
    )
    return SpikingViT1D(config)
