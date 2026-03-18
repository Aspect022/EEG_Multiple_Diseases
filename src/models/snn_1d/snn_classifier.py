"""
SNN 1D Classifier for raw EEG signals.

Two variants:
  - SNN1D_LIF: Conv1d + LIF neurons (no attention) — baseline
  - SNN1D_Attention: Conv1d + LIF + MultiScaleSpikingAttention

Ported from ECG project's ClassicalPath with EEG dimensions:
  Input:  (batch, 6, 3000)  — 6 EEG channels, 30s @ 100Hz
  Output: (batch, 5)        — 5 sleep stage classes

Three-stage architecture:
  Stage 1: Local patterns (spindles, K-complexes) — high resolution
  Stage 2: Segment patterns (sleep cycles) — mid resolution
  Stage 3: Epoch patterns (overall stage) — low resolution
  → Temporal Fusion → Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .lif_neuron import LIFNeuron, LIFLayer
from .attention import MultiScaleSpikingAttention, GlobalSpikingAttention


# ──────────────────────── Depthwise Separable Conv1d ────────────────────────

class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable 1D convolution.
    More efficient than standard conv: depthwise (per-channel) + pointwise (mixing).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


# ──────────────────────── Stage Blocks ────────────────────────

class LocalPatternStage(nn.Module):
    """
    Stage 1: Local high-resolution pattern extractor.
    Captures fine details like sleep spindles (12–14 Hz) and K-complexes.

    Input:  (batch, 6, 3000)
    Output: (batch, 128, 3000), reg_loss
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 128,
        kernel_size: int = 3,
        timesteps: int = 25,  # Increased from 4 to 25 for proper temporal dynamics
    ):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size)
        self.lif_layer = LIFLayer(out_channels)
        self.timesteps = timesteps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x, reg_loss = self.lif_layer(x, timesteps=self.timesteps)
        return x, reg_loss


class SegmentPatternStage(nn.Module):
    """
    Stage 2: Mid-resolution segment pattern extractor.
    Captures patterns spanning ~1-5 seconds (sleep stage sub-patterns).

    Input:  (batch, 128, 3000)
    Output: (batch, 64, 1500), reg_loss
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 64,
        kernel_size: int = 9,
        stride: int = 2,
        timesteps: int = 25,  # Increased from 4 to 25 for proper temporal dynamics
        use_attention: bool = False,
    ):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(
            in_channels, out_channels, kernel_size, stride=stride,
        )
        self.lif_layer = LIFLayer(out_channels)
        self.timesteps = timesteps

        self.use_attention = use_attention
        if use_attention:
            self.mssa = MultiScaleSpikingAttention(
                dim=out_channels,
                local_window=32,
                regional_window=128,
                global_pool=8,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x, reg_loss = self.lif_layer(x, timesteps=self.timesteps)
        if self.use_attention:
            x = self.mssa(x)
        return x, reg_loss


class EpochPatternStage(nn.Module):
    """
    Stage 3: Low-resolution epoch-level pattern extractor.
    Captures overall sleep stage characteristics across the full epoch.

    Input:  (batch, 64, 1500)
    Output: (batch, 32, 750), reg_loss
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 32,
        kernel_size: int = 15,
        stride: int = 2,
        timesteps: int = 25,  # Increased from 4 to 25 for proper temporal dynamics
        use_attention: bool = False,
    ):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(
            in_channels, out_channels, kernel_size, stride=stride,
        )
        self.lif_layer = LIFLayer(out_channels)
        self.timesteps = timesteps

        self.use_attention = use_attention
        if use_attention:
            self.global_attn = GlobalSpikingAttention(out_channels, num_heads=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x, reg_loss = self.lif_layer(x, timesteps=self.timesteps)
        if self.use_attention:
            x = self.global_attn(x)
        return x, reg_loss


# ──────────────────────── Temporal Fusion ────────────────────────

class TemporalFusionBlock(nn.Module):
    """
    Fuse multi-scale features from all 3 stages.
    
    Pools each stage to a common length, concatenates, and projects
    to final embedding dimension.
    
    Inputs: stage1 (128, 3000), stage2 (64, 1500), stage3 (32, 750)
    Output: (batch, fusion_dim)
    """
    
    def __init__(
        self,
        stage1_channels: int = 128,
        stage2_channels: int = 64,
        stage3_channels: int = 32,
        fusion_dim: int = 128,
        pool_length: int = 128,
    ):
        super().__init__()
        total_channels = stage1_channels + stage2_channels + stage3_channels
        
        self.pool = nn.AdaptiveAvgPool1d(pool_length)
        self.alpha = nn.Parameter(torch.ones(3) / 3.0)
        
        self.proj = nn.Conv1d(total_channels, fusion_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(fusion_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(
        self,
        stage1: torch.Tensor,
        stage2: torch.Tensor,
        stage3: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            stage1: (batch, 128, 3000)
            stage2: (batch, 64, 1500)
            stage3: (batch, 32, 750)
        Returns:
            (batch, fusion_dim)
        """
        weights = F.softmax(self.alpha, dim=0)
        
        s1 = self.pool(stage1) * weights[0]
        s2 = self.pool(stage2) * weights[1]
        s3 = self.pool(stage3) * weights[2]
        
        fused = torch.cat([s1, s2, s3], dim=1)  # (batch, 224, pool_length)
        fused = self.proj(fused)
        fused = self.bn(fused)
        fused = F.elu(fused)
        
        output = self.global_pool(fused).squeeze(-1)  # (batch, fusion_dim)
        return output


# ──────────────────────── Full Models ────────────────────────

class SNN1D(nn.Module):
    """
    Spiking Neural Network for 1D EEG signals.

    3-stage hierarchical temporal pyramid with LIF neurons.

    Args:
        in_channels: Number of EEG channels (default 6)
        num_classes: Number of output classes (default 5)
        use_attention: If True, adds spiking attention at stages 2 & 3
        fusion_dim: Final feature dimension before classifier
        timesteps: LIF simulation timesteps (increased from 8 to 25)
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 5,
        use_attention: bool = False,
        fusion_dim: int = 128,
        timesteps: int = 25,  # Increased from 8 to 25 for proper temporal dynamics
    ):
        super().__init__()
        self.use_attention = use_attention
        self.timesteps = timesteps
        self.spike_stats = {}  # For monitoring spike rates during training

        # 3-stage pyramid
        self.stage1 = LocalPatternStage(
            in_channels, out_channels=128, timesteps=timesteps,
        )
        self.stage2 = SegmentPatternStage(
            128, out_channels=64, timesteps=timesteps,
            use_attention=use_attention,
        )
        self.stage3 = EpochPatternStage(
            64, out_channels=32, timesteps=timesteps,
            use_attention=use_attention,
        )

        # Temporal fusion
        self.fusion = TemporalFusionBlock(128, 64, 32, fusion_dim=fusion_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Poisson spike encoding for proper temporal dynamics.
        
        Args:
            x: (batch, 6, 3000) — raw EEG signal
        Returns:
            (batch, num_classes) logits
        """
        # === Handle channel mismatch ===
        # Dataset may provide fewer channels; pad with zeros
        if x.shape[1] < 6:
            pad = torch.zeros(
                x.shape[0], 6 - x.shape[1], x.shape[2],
                device=x.device, dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)

        # === Poisson Spike Encoding ===
        # Convert continuous input to spike probability via rate coding
        # This creates temporal variation essential for SNN computation
        B, C, L = x.shape
        
        # Normalize to [0, 1] range per sample
        x_min = x.view(B, -1).min(dim=1, keepdim=True)[0]
        x_max = x.view(B, -1).max(dim=1, keepdim=True)[0]
        x_norm = (x - x_min.view(-1, 1, 1)) / (x_max.view(-1, 1, 1) - x_min.view(-1, 1, 1) + 1e-8)
        
        # Scale to reasonable firing rate (max ~30% to prevent saturation)
        x_spike_prob = x_norm * 0.3
        
        # Initialize spike statistics monitoring
        self.spike_stats = {'stage_rates': [], 'total_spikes': 0}
        
        # Process through 3-stage pyramid with Poisson encoding
        # Note: The actual spike generation happens inside LIFLayer over timesteps
        # Here we provide the encoded input
        s1, reg1 = self.stage1(x_spike_prob)    # (B, 128, 3000)
        self.spike_stats['stage_rates'].append(s1.mean().item())
        
        s2, reg2 = self.stage2(s1)   # (B, 64, 1500)
        self.spike_stats['stage_rates'].append(s2.mean().item())
        
        s3, reg3 = self.stage3(s2)   # (B, 32, 750)
        self.spike_stats['stage_rates'].append(s3.mean().item())

        # Temporal fusion
        features = self.fusion(s1, s2, s3)  # (B, 128)

        # Classification
        logits = self.classifier(features)

        # Store reg_loss for training (accessed via model.reg_loss)
        self._reg_loss = reg1 + reg2 + reg3
        
        # Store average firing rate for monitoring
        self.spike_stats['avg_rate'] = sum(self.spike_stats['stage_rates']) / len(self.spike_stats['stage_rates'])

        return logits
    
    @property
    def reg_loss(self) -> torch.Tensor:
        """Spike regularization loss from last forward pass."""
        return getattr(self, '_reg_loss', torch.tensor(0.0))
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────── Factory Functions ────────────────────────

def create_snn_1d_lif(num_classes: int = 5, in_channels: int = 6) -> SNN1D:
    """Create SNN-1D-LIF model (no attention)."""
    return SNN1D(
        in_channels=in_channels,
        num_classes=num_classes,
        use_attention=False,
    )


def create_snn_1d_attention(num_classes: int = 5, in_channels: int = 6) -> SNN1D:
    """Create SNN-1D-Attention model (with multi-scale spiking attention)."""
    return SNN1D(
        in_channels=in_channels,
        num_classes=num_classes,
        use_attention=True,
    )
