"""
Multi-Scale Spiking Attention for 1D EEG signals.

Ported from ECG project's attention.py.
Adapted window sizes for EEG (6 channels, 3000 samples @ 100Hz).

Three attention scales:
- Local Window: Fine-grained (sleep spindles, K-complexes)
- Regional Dilated: Medium-range (sleep stage transitions)
- Global Pooled: Full-epoch overview (overall sleep stage)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LocalWindowAttention(nn.Module):
    """
    Local windowed attention for fine-grained pattern capture.
    
    Captures local patterns like sleep spindles (0.5–2s at 100Hz = 50–200 samples).
    
    Args:
        dim: Feature dimension
        window_size: Size of attention window (default 32 for EEG)
        num_heads: Number of attention heads
    """
    
    def __init__(self, dim: int, window_size: int = 32, num_heads: int = 1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias
        self.relative_position_bias = nn.Parameter(
            torch.zeros(2 * window_size - 1)
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, length, dim)
        Returns:
            (batch, length, dim)
        """
        B, L, C = x.shape
        
        # Pad for window division
        pad_len = (self.window_size - L % self.window_size) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        padded_L = x.shape[1]
        num_windows = padded_L // self.window_size
        
        # Reshape into windows: (B, num_windows, window_size, C)
        x = x.view(B, num_windows, self.window_size, C)
        
        # QKV
        qkv = self.qkv(x).reshape(
            B, num_windows, self.window_size, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)  # (3, B, nW, heads, ws, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        position_ids = torch.arange(self.window_size, device=x.device)
        relative_positions = position_ids.unsqueeze(0) - position_ids.unsqueeze(1)
        relative_positions = relative_positions + self.window_size - 1
        rel_bias = self.relative_position_bias[relative_positions]
        attn = attn + rel_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(2, 3).reshape(
            B, num_windows, self.window_size, C
        )
        out = self.proj(out)
        out = out.view(B, padded_L, C)
        
        if pad_len > 0:
            out = out[:, :L, :]
        
        return out


class RegionalDilatedAttention(nn.Module):
    """
    Regional dilated attention for medium-scale pattern capture.
    
    Uses dilated sampling to capture patterns across sleep cycles.
    
    Args:
        dim: Feature dimension
        window_size: Base window size
        dilation: Dilation factor
    """
    
    def __init__(self, dim: int, window_size: int = 128, dilation: int = 4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.dilation = dilation
        self.scale = dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, length, dim)
        Returns:
            (batch, length, dim)
        """
        B, L, C = x.shape
        
        # Sample at dilated positions
        sampled_indices = torch.arange(0, L, self.dilation, device=x.device)
        x_sampled = x[:, sampled_indices, :]  # (B, L//d, C)
        
        # QKV on sampled sequence
        qkv = self.qkv(x_sampled).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Global attention on sampled sequence
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out_sampled = attn @ v
        out_sampled = self.proj(out_sampled)
        
        # Interpolate back to original length
        out_sampled = out_sampled.transpose(1, 2)  # (B, C, L//d)
        out = F.interpolate(out_sampled, size=L, mode='linear', align_corners=False)
        out = out.transpose(1, 2)  # (B, L, C)
        
        return out


class GlobalPooledAttention(nn.Module):
    """
    Lightweight global attention using pooled keys/values.
    
    Captures overall epoch-level patterns without quadratic cost.
    
    Args:
        dim: Feature dimension
        pool_size: Pooling factor for K/V
    """
    
    def __init__(self, dim: int, pool_size: int = 8):
        super().__init__()
        self.dim = dim
        self.pool_size = pool_size
        self.scale = dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, length, dim)
        Returns:
            (batch, length, dim)
        """
        B, L, C = x.shape
        
        q = self.q_proj(x)  # (B, L, C)
        
        # Pooled keys and values
        x_t = x.transpose(1, 2)  # (B, C, L)
        x_pooled = self.pool(x_t).transpose(1, 2)  # (B, pool_size, C)
        
        k = self.k_proj(x_pooled)
        v = self.v_proj(x_pooled)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = attn @ v
        out = self.proj(out)
        
        return out


class MultiScaleSpikingAttention(nn.Module):
    """
    Multi-Scale Spiking Attention (MSSA) for EEG signals.
    
    Combines local, regional, and global attention with learned weights.
    Adapted from ECG MSSA with EEG-appropriate window sizes.
    
    Args:
        dim: Feature dimension
        local_window: Window size for local attention (default 32)
        regional_window: Window for regional attention (default 128)
        global_pool: Pool size for global attention (default 8)
    """
    
    def __init__(
        self,
        dim: int,
        local_window: int = 32,
        regional_window: int = 128,
        global_pool: int = 8,
    ):
        super().__init__()
        self.dim = dim
        
        # Three attention scales
        self.local_attn = LocalWindowAttention(dim, local_window, num_heads=1)
        self.regional_attn = RegionalDilatedAttention(dim, regional_window, dilation=4)
        self.global_attn = GlobalPooledAttention(dim, global_pool)
        
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(3) / 3.0)
        
        # Norms and FFN
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1),
        )
        self.ffn_norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, length) — conv format
        Returns:
            (batch, channels, length)
        """
        # Transpose for attention: (B, L, C)
        x = x.transpose(1, 2)
        
        # Multi-scale attention with residual
        residual = x
        x = self.norm(x)
        
        weights = F.softmax(self.scale_weights, dim=0)
        
        local_out = self.local_attn(x)
        regional_out = self.regional_attn(x)
        global_out = self.global_attn(x)
        
        attn_out = (
            weights[0] * local_out
            + weights[1] * regional_out
            + weights[2] * global_out
        )
        x = residual + attn_out
        
        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        
        # Back to (B, C, L)
        return x.transpose(1, 2)


class GlobalSpikingAttention(nn.Module):
    """
    Simple global attention for rhythm-level patterns.
    Lightweight version for Stage 3.
    """
    
    def __init__(self, dim: int, num_heads: int = 1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, length)
        Returns:
            (batch, channels, length)
        """
        x = x.transpose(1, 2)  # (B, L, C)
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        x = residual + x
        return x.transpose(1, 2)
