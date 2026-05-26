import torch
import torch.nn as nn
import torch.nn.functional as F
from ..qspikexai_net import TASK_CHANNELS, TASK_SEQ_LEN
from .vit1d import TransformerEncoderLayer1D

class EEG_MFTNet(nn.Module):
    """
    EEG-MFTNet: Multi-scale Feature Temporal Network + Transformer.
    Uses parallel convolutions with different kernel sizes to extract multi-scale
    temporal features, followed by a Transformer encoder.
    """

    def __init__(
        self,
        task: str,
        f1: int = 16,
        dim: int = 128,
        depth: int = 2,
        heads: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        self.task = task
        self.in_channels = TASK_CHANNELS[task]
        self.seq_len = TASK_SEQ_LEN[task]
        
        # Parallel convolutions with kernel sizes 5, 9, 13, 29, 61, 125
        self.kernels = [5, 9, 13, 29, 61, 125]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.in_channels, f1, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(f1),
                nn.ELU(),
                nn.Dropout(dropout)
            ) for k in self.kernels
        ])
        
        # Project combined multi-scale features to transformer dimension
        total_conv_features = f1 * len(self.kernels)
        self.proj = nn.Linear(total_conv_features, dim)
        
        # Transformer backend
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len // 4 + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.transformer = nn.Sequential(*[
            TransformerEncoderLayer1D(dim, heads, dim * 2, dropout)
            for _ in range(depth)
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 if task == 'sleep_apnea' else 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        batch_size = x.shape[0]
        
        # Extract multi-scale features
        conv_outs = [conv(x) for conv in self.convs]
        h = torch.cat(conv_outs, dim=1) # (B, total_conv_features, T)
        
        # Downsample time resolution by 4 for efficiency (pooling)
        h = F.avg_pool1d(h, kernel_size=4) # (B, total_conv_features, T // 4)
        T_reduced = h.shape[-1]
        
        # Reshape for transformer: (B, T // 4, total_conv_features)
        h = h.permute(0, 2, 1)
        h = self.proj(h) # (B, T // 4, dim)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1) # (B, T // 4 + 1, dim)
        
        # Add position embeddings
        if h.shape[1] <= self.pos_embedding.shape[1]:
            pos = self.pos_embedding[:, :h.shape[1]]
        else:
            # Dynamically interpolate pos embedding if needed
            pos = F.interpolate(self.pos_embedding.permute(0, 2, 1), size=h.shape[1], mode='linear', align_corners=True).permute(0, 2, 1)
            
        h = h + pos
        
        # Apply transformer
        h = self.transformer(h)
        
        # Classify CLS token
        cls_out = h[:, 0]
        return self.classifier(cls_out)
