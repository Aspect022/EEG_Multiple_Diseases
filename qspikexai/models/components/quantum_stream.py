import torch
import torch.nn as nn
import torch.nn.functional as F
from .vqc_layer import VQCLayer

class QuantumStream(nn.Module):
    """
    Quantum-enhanced EEG spectral stream.
    Input:  (B, n_channels, n_freqs=40, n_times)
    Output: (B, 128)  — feature vector
    """

    def __init__(self, n_channels: int, n_freqs: int = 40,
                 n_qubits: int = 4, n_heads: int = 4):
        super().__init__()
        self.n_heads   = n_heads
        self.dim_head  = 2 ** n_qubits   # 16

        # Convolutional patch encoder (replaces pure linear ViT patch embedding)
        # Reduces spectral/temporal dimension to 64 features
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=(4, 4), stride=(4, 4)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),   # (B, 64, 1, 1)
        )

        # Classical → quantum state encoding mapping
        self.to_quantum = nn.Linear(64, n_heads * self.dim_head)

        # Shared VQC across heads for parameter efficiency
        self.vqc = VQCLayer(n_qubits=n_qubits)

        # Post-measurement projection
        self.post_proj = nn.Linear(n_heads * self.dim_head, 128)
        self.norm      = nn.LayerNorm(128)
        self.dropout   = nn.Dropout(0.3)

    def forward(self, scalogram: torch.Tensor) -> torch.Tensor:
        # scalogram: (B, C, F, T)
        B = scalogram.shape[0]

        # 1. Patch encoding
        h = self.patch_encoder(scalogram)       # (B, 64, 1, 1)
        h = h.view(B, -1)                        # (B, 64)

        # 2. Map to quantum state amplitudes
        q = self.to_quantum(h)                   # (B, n_heads * dim_head)
        q = q.view(B * self.n_heads, self.dim_head)
        q = F.normalize(q, p=2, dim=-1)          # normalize to unit sphere (real amplitudes)

        # 3. Encode as complex state vector
        state = torch.complex(q, torch.zeros_like(q)).unsqueeze(-1)  # (B*H, dim, 1)

        # 4. Variational Quantum Circuit (VQC)
        state = self.vqc(state)                  # (B*H, dim, 1)

        # 5. Measurement: Born rule expectation (|ψ|²)
        meas = state.squeeze(-1).abs() ** 2      # (B*H, dim) — real probability values
        meas = meas.view(B, self.n_heads * self.dim_head)   # (B, H*dim)

        # 6. Project back to classical space
        out = self.post_proj(meas)               # (B, 128)
        return self.dropout(self.norm(out))
