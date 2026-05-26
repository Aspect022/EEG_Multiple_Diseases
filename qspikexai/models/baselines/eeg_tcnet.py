import torch
import torch.nn as nn
from ..qspikexai_net import TASK_CHANNELS, TASK_SEQ_LEN

class CausalConv1d(nn.Module):
    """Causal 1D Convolution wrapper to preserve temporal causality."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad on the left side only for causality
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)

import torch.nn.functional as F

class TCNBlock(nn.Module):
    """Temporal Convolutional Network residual block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        h = F.elu(self.bn1(self.conv1(x)))
        h = self.dropout(h)
        h = self.bn2(self.conv2(h))
        h = self.dropout(h)
        return F.elu(h + res)

class EEGTCNet(nn.Module):
    """
    EEG-TCNet: EEGNet + Temporal Convolutional Network.
    Reference: Ingolfsson et al. 2020 (https://arxiv.org/abs/2006.00622)
    """

    def __init__(
        self,
        task: str,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        tcn_filters: int = 12,
        tcn_depth: int = 2,
        kernel_size: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        self.task = task
        self.in_channels = TASK_CHANNELS[task]
        self.seq_len = TASK_SEQ_LEN[task]
        
        # Block 1: EEGNet-like frontend (extract spatial-temporal features)
        self.frontend = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(self.in_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F2, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout)
        )
        
        # TCN backend
        tcn_layers = []
        in_ch = F2
        for i in range(tcn_depth):
            dilation = 2 ** i
            tcn_layers.append(TCNBlock(in_ch, tcn_filters, kernel_size, dilation, dropout))
            in_ch = tcn_filters
            
        self.tcn = nn.Sequential(*tcn_layers)
        
        # Classifier
        self.time_reduced = self.seq_len // 32
        self.flat_features = tcn_filters * self.time_reduced
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_features, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4 if task == 'sleep_apnea' else 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)
        h = self.frontend(x)
        
        # (B, F2, 1, T') -> (B, F2, T')
        h = h.squeeze(2)
        
        # Ensure temporal dimension matches expected size
        if h.shape[-1] != self.time_reduced:
            h = F.adaptive_avg_pool1d(h, self.time_reduced)
            
        h = self.tcn(h)
        return self.classifier(h)
