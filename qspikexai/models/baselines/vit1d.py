import torch
import torch.nn as nn
import torch.nn.functional as F
from ..qspikexai_net import TASK_CHANNELS, TASK_SEQ_LEN

class TransformerEncoderLayer1D(nn.Module):
    def __init__(self, dim: int, heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN structure
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h)
        x = x + self.dropout1(attn_out)
        
        h = self.norm2(x)
        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(h))))
        x = x + self.dropout2(ff_out)
        return x

class ViT1D(nn.Module):
    """
    1D Vision Transformer for EEG classification.
    Segments time series of each channel into patches.
    """

    def __init__(
        self,
        task: str,
        patch_size: int = 64,
        dim: int = 128,
        depth: int = 4,
        heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.task = task
        self.in_channels = TASK_CHANNELS[task]
        self.seq_len = TASK_SEQ_LEN[task]
        self.patch_size = patch_size
        
        assert self.seq_len % patch_size == 0, f"Sequence length {self.seq_len} must be divisible by patch size {patch_size}"
        self.num_patches = self.seq_len // patch_size
        
        # Patch projection (projection per channel patch)
        self.patch_to_embedding = nn.Linear(patch_size * self.in_channels, dim)
        
        # Class token and Positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.transformer = nn.Sequential(*[
            TransformerEncoderLayer1D(dim, heads, dim_feedforward, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 4 if task == 'sleep_apnea' else 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        B, C, T = x.shape
        
        # Reshape to patches: (B, C, num_patches, patch_size) -> permute to (B, num_patches, C, patch_size)
        x_patches = x.view(B, C, self.num_patches, self.patch_size)
        x_patches = x_patches.permute(0, 2, 1, 3).reshape(B, self.num_patches, -1)
        
        # Project patches to dim
        x_embeddings = self.patch_to_embedding(x_patches)  # (B, num_patches, dim)
        
        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x_embeddings = torch.cat((cls_tokens, x_embeddings), dim=1)  # (B, num_patches + 1, dim)
        
        # Add positional embeddings
        x_embeddings = x_embeddings + self.pos_embedding
        x_embeddings = self.dropout(x_embeddings)
        
        # Process transformer
        h = self.transformer(x_embeddings)
        h = self.norm(h)
        
        # Classification head on CLS token
        cls_out = h[:, 0]
        return self.classifier(cls_out)
