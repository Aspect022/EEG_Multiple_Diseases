import torch
import torch.nn as nn
import torch.nn.functional as F

class SurrogateGradient(torch.autograd.Function):
    """Fast sigmoid surrogate gradient for spiking threshold."""
    @staticmethod
    def forward(ctx, mem, threshold):
        ctx.save_for_backward(mem, threshold)
        return (mem >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        mem, threshold = ctx.saved_tensors
        # Derivative of sigmoid(4 * x) is 4 * sig(x) * (1 - sig(x))
        x = mem - threshold
        sig = torch.sigmoid(4.0 * x)
        grad_input = grad_output * sig * (1.0 - sig) * 4.0
        return grad_input, None

class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with learnable parameters."""
    def __init__(self, tau_mem: float = 0.9, threshold: float = 1.0):
        super().__init__()
        # Map tau_mem to raw_tau such that sigmoid(raw_tau) * 0.85 + 0.1 = tau_mem
        # For tau_mem=0.9: raw_tau = log(0.8 / 0.05) = log(16) ≈ 2.77
        self.raw_tau = nn.Parameter(torch.tensor(2.77))
        self.threshold = nn.Parameter(torch.tensor(threshold))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        # Differentiable tau mapping between [0.1, 0.95]
        tau = torch.sigmoid(self.raw_tau) * 0.85 + 0.1
        mem = torch.zeros_like(x[..., 0])
        spikes = []
        for t in range(x.shape[-1]):
            mem = tau * mem + (1.0 - tau) * x[..., t]
            spike = SurrogateGradient.apply(mem, self.threshold)
            mem = mem * (1.0 - spike)  # Soft reset
            spikes.append(spike)
        return torch.stack(spikes, dim=-1)   # (B, C, T)

class SNN1DAttentionStream(nn.Module):
    """
    1D Spiking Neural Network with temporal attention.
    Input:  (B, n_channels, T)
    Output: (B, 256)  — feature vector
    """
    def __init__(self, n_channels: int, seq_len: int, hidden: int = 256):
        super().__init__()
        # Temporal convolutional feature extraction
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=25, padding=12, stride=2),
            nn.BatchNorm1d(64),
            LIFNeuron(),
            nn.Conv1d(64, 128, kernel_size=15, padding=7, stride=2),
            nn.BatchNorm1d(128),
            LIFNeuron(),
            nn.Conv1d(128, 256, kernel_size=9, padding=4, stride=2),
            nn.BatchNorm1d(256),
            LIFNeuron(),
        )
        # Temporal self-attention
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(256)
        # Pooling to fixed size
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(256, hidden)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.conv_block(x)              # (B, 256, T')
        h = h.permute(0, 2, 1)             # (B, T', 256) for attention
        attn_out, _ = self.attn(h, h, h)   # (B, T', 256)
        h = self.attn_norm(h + attn_out)    # residual
        h = h.permute(0, 2, 1)             # (B, 256, T')
        h = self.pool(h).squeeze(-1)        # (B, 256)
        return self.dropout(F.gelu(self.proj(h)))
