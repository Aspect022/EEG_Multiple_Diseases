"""
Gated Fusion primitives for combining multiple model streams.

Ported from ECG project's fusion.py:
- GatedFusionModule: 2-way fusion with learned gate g ∈ [0,1]
- MultiStreamFusion: N-way extension with per-stream gates
- ClassificationHead: MLP classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class GatedFusionModule(nn.Module):
    """
    Gated Fusion of two feature streams.
    
    Learns a gate g ∈ [0, 1]:
        h_fused = (1 - g) * h_primary + g * h_secondary_expanded
    
    Args:
        primary_dim: Dimension of primary stream
        secondary_dim: Dimension of secondary stream
        gate_hidden: Hidden dim for gate network
    """
    
    def __init__(
        self,
        primary_dim: int,
        secondary_dim: int,
        gate_hidden: int = 64,
    ):
        super().__init__()
        
        # Expand secondary to match primary dimension
        self.secondary_expansion = nn.Sequential(
            nn.Linear(secondary_dim, gate_hidden),
            nn.ELU(),
            nn.Linear(gate_hidden, primary_dim),
        )
        
        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(primary_dim + secondary_dim, gate_hidden),
            nn.ELU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )
        
        self.norm = nn.LayerNorm(primary_dim)
    
    def forward(
        self,
        primary: torch.Tensor,
        secondary: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            primary: (batch, primary_dim)
            secondary: (batch, secondary_dim)
        Returns:
            (fused_features, gate_value)
        """
        combined = torch.cat([primary, secondary], dim=1)
        gate = self.gate_network(combined)  # (batch, 1)
        
        secondary_expanded = self.secondary_expansion(secondary)
        
        fused = (1 - gate) * primary + gate * secondary_expanded
        fused = self.norm(fused)
        
        return fused, gate


class MultiStreamFusion(nn.Module):
    """
    N-way fusion of multiple feature streams.
    
    Each stream is projected to a common dimension, then combined
    with learned attention weights.
    
    Args:
        stream_dims: List of input dims for each stream
        fusion_dim: Common output dimension
    """
    
    def __init__(self, stream_dims: List[int], fusion_dim: int = 768):
        super().__init__()
        
        self.n_streams = len(stream_dims)
        
        # Project each stream to fusion_dim
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ELU(),
            )
            for dim in stream_dims
        ])
        
        # Attention weights for combining streams
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * self.n_streams, self.n_streams),
            nn.Softmax(dim=-1),
        )
        
        self.norm = nn.LayerNorm(fusion_dim)
    
    def forward(self, streams: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            streams: List of tensors, each (batch, stream_dim_i)
        Returns:
            (batch, fusion_dim)
        """
        assert len(streams) == self.n_streams
        
        # Project all streams
        projected = [proj(s) for proj, s in zip(self.projections, streams)]
        
        # Compute attention weights
        concat = torch.cat(projected, dim=1)  # (batch, fusion_dim * n)
        weights = self.attention(concat)       # (batch, n_streams)
        
        # Weighted combination
        stacked = torch.stack(projected, dim=1)  # (batch, n, fusion_dim)
        weights = weights.unsqueeze(-1)           # (batch, n, 1)
        fused = (stacked * weights).sum(dim=1)    # (batch, fusion_dim)
        
        return self.norm(fused)


class ClassificationHead(nn.Module):
    """
    MLP classification head.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
