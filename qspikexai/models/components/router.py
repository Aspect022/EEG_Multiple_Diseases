import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalRouter(nn.Module):
    """
    Confidence-Gated Conditional Router.
    Learns to dynamically weight SNN (temporal) and QNN (spectral) streams.
    
    Input:  F_snn (B, snn_dim), F_qnn (B, qnn_dim)
    Output: F_fused (B, fused_dim) and gate (B, 2)
    """

    def __init__(self, snn_dim: int = 256, qnn_dim: int = 128, fused_dim: int = 256):
        super().__init__()
        concat_dim = snn_dim + qnn_dim    # 384

        # Gate network to estimate confidence weights
        self.gate_net = nn.Sequential(
            nn.Linear(concat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

        # Projection layers to map streams to unified dimension
        self.proj_snn = nn.Linear(snn_dim, fused_dim)
        self.proj_qnn = nn.Linear(qnn_dim, fused_dim)

    def forward(self, f_snn: torch.Tensor, f_qnn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat([f_snn, f_qnn], dim=-1)     # (B, 384)
        gate   = F.softmax(self.gate_net(concat), dim=-1)   # (B, 2)

        alpha_snn = gate[:, 0:1]   # (B, 1)
        alpha_qnn = gate[:, 1:2]   # (B, 1)

        # Confidence blend
        fused = alpha_snn * self.proj_snn(f_snn) + alpha_qnn * self.proj_qnn(f_qnn)   # (B, fused_dim)

        return fused, gate   # Return gate for XAI logging
