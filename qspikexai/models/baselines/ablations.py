import torch
import torch.nn as nn
import torch.nn.functional as F
from ..components.snn_stream import SNN1DAttentionStream
from ..components.quantum_stream import QuantumStream
from ..components.task_heads import build_task_head
from ...utils.transforms import batch_cwt

class SNNOnlyModel(nn.Module):
    """SNN-1D-Attention stream only (Ablation)."""
    def __init__(self, task: str):
        super().__init__()
        self.task = task
        # Channel counts and sequence lengths are imported/handled internally
        from ..qspikexai_net import TASK_CHANNELS, TASK_SEQ_LEN
        n_channels = TASK_CHANNELS[task]
        seq_len = TASK_SEQ_LEN[task]
        
        self.snn = SNN1DAttentionStream(n_channels=n_channels, seq_len=seq_len, hidden=256)
        self.head = build_task_head(task, fused_dim=256)
        
    def forward(self, x_raw: torch.Tensor, x_scalogram: torch.Tensor = None) -> torch.Tensor:
        # Ignore scalogram
        f_snn = self.snn(x_raw)
        return self.head(f_snn)

class QuantumOnlyModel(nn.Module):
    """Quantum-only stream (Ablation)."""
    def __init__(self, task: str):
        super().__init__()
        self.task = task
        from ..qspikexai_net import TASK_CHANNELS
        n_channels = TASK_CHANNELS[task]
        
        self.qnn = QuantumStream(n_channels=n_channels, n_freqs=40, n_qubits=4, n_heads=4)
        self.head = build_task_head(task, fused_dim=128)
        
    def forward(self, x_raw: torch.Tensor, x_scalogram: torch.Tensor = None) -> torch.Tensor:
        if x_scalogram is None:
            x_scalogram = batch_cwt(x_raw)
        f_qnn = self.qnn(x_scalogram)
        return self.head(f_qnn)

class ConcatFusionModel(nn.Module):
    """Dual-stream fusion via fixed concatenation without gating router (Ablation)."""
    def __init__(self, task: str):
        super().__init__()
        self.task = task
        from ..qspikexai_net import TASK_CHANNELS, TASK_SEQ_LEN
        n_channels = TASK_CHANNELS[task]
        seq_len = TASK_SEQ_LEN[task]
        
        self.snn = SNN1DAttentionStream(n_channels=n_channels, seq_len=seq_len, hidden=256)
        self.qnn = QuantumStream(n_channels=n_channels, n_freqs=40, n_qubits=4, n_heads=4)
        
        # Concat dim = 256 + 128 = 384
        self.head = build_task_head(task, fused_dim=384)
        
    def forward(self, x_raw: torch.Tensor, x_scalogram: torch.Tensor = None) -> torch.Tensor:
        if x_scalogram is None:
            x_scalogram = batch_cwt(x_raw)
            
        f_snn = self.snn(x_raw)
        f_qnn = self.qnn(x_scalogram)
        
        f_fused = torch.cat([f_snn, f_qnn], dim=-1)
        return self.head(f_fused)
