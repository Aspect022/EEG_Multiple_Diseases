import torch
import torch.nn as nn
from .components.snn_stream     import SNN1DAttentionStream
from .components.quantum_stream import QuantumStream
from .components.router         import ConditionalRouter
from .components.task_heads     import build_task_head
from ..utils.transforms         import batch_cwt

# Per-task channel counts
TASK_CHANNELS = {
    'sleep_apnea':   1,
    'schizophrenia': 19,
    'mci':           19,
    'depression':    19,
}

# Per-task sequence lengths (at 256 Hz, after windowing)
TASK_SEQ_LEN = {
    'sleep_apnea':   15360,
    'schizophrenia': 2560,
    'mci':           2048,
    'depression':    2048,
}

class QSpikeXAINet(nn.Module):
    """
    QSpikeXAI-Net: Main proposed model.
    Composes SNN and QNN parallel streams with conditional routing.
    
    Args:
        task: one of 'sleep_apnea' | 'schizophrenia' | 'mci' | 'depression'
        n_qubits: number of qubits for VQC (default 4)
        n_heads_vqc: number of parallel VQC heads (default 4)
    """

    def __init__(self, task: str, n_qubits: int = 4, n_heads_vqc: int = 4):
        super().__init__()
        self.task       = task
        n_channels      = TASK_CHANNELS[task]
        seq_len         = TASK_SEQ_LEN[task]

        # Stream 1: SNN temporal (1D signal input)
        self.snn_stream = SNN1DAttentionStream(
            n_channels=n_channels,
            seq_len=seq_len,
            hidden=256
        )

        # Stream 2: Quantum spectral (CWT scalogram input)
        self.quantum_stream = QuantumStream(
            n_channels=n_channels,
            n_freqs=40,
            n_qubits=n_qubits,
            n_heads=n_heads_vqc
        )

        # Router
        self.router = ConditionalRouter(snn_dim=256, qnn_dim=128, fused_dim=256)

        # Task specific classification head
        self.task_head = build_task_head(task, fused_dim=256)

    def forward(self, x_raw: torch.Tensor, x_scalogram: torch.Tensor = None, return_gate: bool = False):
        """
        Args:
            x_raw:       (B, n_channels, T) — raw EEG signal
            x_scalogram: (B, n_channels, 40, T') — CWT scalogram
                         if None, compute on-the-fly (slower)
            return_gate: if True, also return gate weights for explainability
        Returns:
            logits: (B, n_classes)
            gate:   (B, 2) — only if return_gate=True
        """
        # Compute CWT scalogram on-the-fly if not provided
        if x_scalogram is None:
            x_scalogram = batch_cwt(x_raw)

        # Process parallel streams
        f_snn = self.snn_stream(x_raw)            # (B, 256)
        f_qnn = self.quantum_stream(x_scalogram)  # (B, 128)

        # Blended representation via the router
        f_fused, gate = self.router(f_snn, f_qnn)  # (B, 256), (B, 2)

        # Final task head predictions
        logits = self.task_head(f_fused)          # (B, n_classes)

        if return_gate:
            return logits, gate
        return logits
