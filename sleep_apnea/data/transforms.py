"""Data transforms for sleep apnea classification."""

from typing import Optional, Tuple
import numpy as np
import torch

try:
    import neurokit2 as nk
except ImportError:
    nk = None


def create_apnea_transform(
    output_size: Tuple[int, int] = (224, 224),
    sampling_rate: float = 100.0,
    normalize: bool = True,
    augment: bool = False,
):
    """
    Create transform pipeline for sleep apnea classification.
    
    Args:
        output_size: Target size for scalogram images
        sampling_rate: Signal sampling rate
        normalize: Apply z-score normalization
        augment: Apply data augmentation
    
    Returns:
        Transform function
    """
    def transform(signal: torch.Tensor) -> torch.Tensor:
        # signal: (channels, time) or (time,)
        
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)  # (1, time)
        
        # Z-score normalization
        if normalize:
            signal = (signal - signal.mean(dim=-1, keepdim=True)) / (signal.std(dim=-1, keepdim=True) + 1e-8)
        
        # TODO: Add scalogram conversion if needed
        # For now, return raw signal
        
        return signal
    
    return transform
