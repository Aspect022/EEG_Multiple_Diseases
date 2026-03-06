"""
Spiking Vision Transformer (Spikformer-inspired) for ECG classification.

All activations are spiking neurons (LIF or QIF). Spiking Self-Attention
uses spike-based Q, K, V projections — no softmax.

References:
    - Zhou et al., "Spikformer", ICLR 2023.
    - Yao et al., "Spike-driven Transformer", NeurIPS 2023.
"""

import torch
import torch.nn as nn
import snntorch as snn
from .spiking_resnet import create_neuron, QuadraticIF


class SpikingPatchEmbedding(nn.Module):
    """Patch embedding with spiking activation."""

    def __init__(self, in_channels=3, embed_dim=256, patch_size=16,
                 neuron_type='lif', beta=0.9):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size, bias=False)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.neuron = create_neuron(neuron_type=neuron_type, beta=beta)

    def forward(self, x):
        x = self.bn(self.proj(x))       # (B, embed_dim, H/P, W/P)
        spk = self.neuron(x)
        B, C, H, W = spk.shape
        return spk.flatten(2).transpose(1, 2)  # (B, N, embed_dim)


class SpikingSelfAttention(nn.Module):
    """Spiking Self-Attention with spike-based Q/K/V (no softmax)."""

    def __init__(self, embed_dim=256, num_heads=4, neuron_type='lif', beta=0.9):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.q_neuron = create_neuron(neuron_type=neuron_type, beta=beta)

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.k_neuron = create_neuron(neuron_type=neuron_type, beta=beta)

        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_bn = nn.BatchNorm1d(embed_dim)
        self.v_neuron = create_neuron(neuron_type=neuron_type, beta=beta)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_neuron(self.q_bn(self.q_proj(x).transpose(1, 2)).transpose(1, 2))
        k = self.k_neuron(self.k_bn(self.k_proj(x).transpose(1, 2)).transpose(1, 2))
        v = self.v_neuron(self.v_bn(self.v_proj(x).transpose(1, 2)).transpose(1, 2))

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(out)


class SpikingMLP(nn.Module):
    """Spiking MLP with two linear layers and spiking activations."""

    def __init__(self, embed_dim=256, mlp_ratio=4.0, neuron_type='lif',
                 beta=0.9, dropout=0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.neuron1 = create_neuron(neuron_type=neuron_type, beta=beta)
        self.fc2 = nn.Linear(hidden, embed_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.neuron2 = create_neuron(neuron_type=neuron_type, beta=beta)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.neuron1(self.bn1(self.fc1(x).transpose(1, 2)).transpose(1, 2))
        out = self.dropout(out)
        out = self.neuron2(self.bn2(self.fc2(out).transpose(1, 2)).transpose(1, 2))
        return self.dropout(out)


class SpikingTransformerBlock(nn.Module):
    """SSA + SpikingMLP with residual connections."""

    def __init__(self, embed_dim=256, num_heads=4, mlp_ratio=4.0,
                 neuron_type='lif', beta=0.9, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SpikingSelfAttention(embed_dim, num_heads, neuron_type, beta)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = SpikingMLP(embed_dim, mlp_ratio, neuron_type, beta, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpikingVisionTransformer(nn.Module):
    """
    Spiking Vision Transformer for ECG scalogram classification.

    Args:
        num_classes: Output classes. Default: 5.
        in_channels: Input channels. Default: 3.
        img_size: Input image size. Default: 224.
        patch_size: Patch size. Default: 16.
        embed_dim: Embedding dim. Default: 256.
        depth: Transformer blocks. Default: 4.
        num_heads: Attention heads. Default: 4.
        num_timesteps: Temporal steps. Default: 25.
        neuron_type: 'lif' or 'qif'. Default: 'lif'.
        beta: LIF decay rate. Default: 0.9.
    """

    def __init__(self, num_classes=5, in_channels=3, img_size=224, patch_size=16,
                 embed_dim=256, depth=4, num_heads=4, mlp_ratio=4.0,
                 num_timesteps=25, neuron_type='lif', beta=0.9, dropout=0.1):
        super().__init__()
        self.num_timesteps = num_timesteps
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = SpikingPatchEmbedding(
            in_channels, embed_dim, patch_size, neuron_type, beta)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.Sequential(*[
            SpikingTransformerBlock(embed_dim, num_heads, mlp_ratio,
                                   neuron_type, beta, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_neuron = create_neuron(neuron_type=neuron_type, beta=beta)

    def _reset_states(self):
        for module in self.modules():
            if isinstance(module, snn.Leaky):
                module.init_leaky()
            elif isinstance(module, QuadraticIF):
                module.reset_mem()

    def forward_timestep(self, x):
        x = self.patch_embed(x) + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x).mean(dim=1)
        return self.head_neuron(self.head(x))

    def forward(self, x):
        self._reset_states()
        spike_record = []
        for t in range(self.num_timesteps):
            spike_record.append(self.forward_timestep(x))
        return torch.stack(spike_record, dim=0).mean(dim=0)


def create_spiking_vit(num_classes=5, num_timesteps=25, neuron_type='lif',
                       variant='small', **kwargs):
    """Factory for Spiking ViT models."""
    configs = {
        'tiny':  {'embed_dim': 128, 'depth': 3, 'num_heads': 4, 'mlp_ratio': 3.0},
        'small': {'embed_dim': 256, 'depth': 4, 'num_heads': 4, 'mlp_ratio': 4.0},
        'base':  {'embed_dim': 384, 'depth': 6, 'num_heads': 6, 'mlp_ratio': 4.0},
    }
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}")
    cfg = configs[variant]
    cfg.update(kwargs)
    return SpikingVisionTransformer(num_classes=num_classes,
                                    num_timesteps=num_timesteps,
                                    neuron_type=neuron_type, **cfg)
