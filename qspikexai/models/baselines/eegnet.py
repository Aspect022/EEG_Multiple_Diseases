import torch
import torch.nn as nn
from ..qspikexai_net import TASK_CHANNELS, TASK_SEQ_LEN
from ..components.task_heads import build_task_head

class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.
    Reference: Lawhern et al. 2018 (https://arxiv.org/abs/1611.08024)
    """

    def __init__(
        self,
        task: str,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5
    ):
        super().__init__()
        self.task = task
        self.in_channels = TASK_CHANNELS[task]
        self.seq_len = TASK_SEQ_LEN[task]
        
        # Block 1: Temporal Conv -> Depthwise Conv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise Conv: filter over channels
            nn.Conv2d(F1, F1 * D, kernel_size=(self.in_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout)
        )
        
        # Block 2: Separable Conv
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F2, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout)
        )
        
        # Calculate features dimension before classifier
        # seq_len -> avg_pool(4) -> avg_pool(8) => seq_len // 32
        self.time_reduced = self.seq_len // 32
        self.flat_features = F2 * self.time_reduced
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_features, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4 if task == 'sleep_apnea' else 2) # maps to task head size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        # Add channel dim: (B, 1, C, T)
        x = x.unsqueeze(1)
        h = self.block1(x)
        h = self.block2(h)
        # Ensure temporal dimension matches flat_features in case division wasn't clean
        if h.shape[-1] != self.time_reduced:
            h = F.adaptive_avg_pool2d(h, (1, self.time_reduced))
        return self.classifier(h)
