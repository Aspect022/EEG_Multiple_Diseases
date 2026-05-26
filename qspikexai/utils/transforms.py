import numpy as np
import torch

def cwt_transform(
    epochs: np.ndarray, 
    fs: float = 256.0, 
    frequencies: np.ndarray = None, 
    wavelet: str = 'morlet'
) -> np.ndarray:
    """
    Continuous Wavelet Transform using standalone NumPy convolution.
    Completely independent of SciPy signal functions to ensure cross-platform compatibility.
    
    Args:
        epochs: numpy array of shape (n_epochs, n_channels, n_samples)
        fs: sampling rate in Hz
        frequencies: array of target frequencies in Hz (default: 0.5 to 45Hz, 40 points)
        wavelet: 'morlet' (default)
        
    Returns:
        scalogram: numpy array of shape (n_epochs, n_channels, n_freqs, n_times_out)
    """
    if frequencies is None:
        frequencies = np.linspace(0.5, 45.0, 40)
        
    n_epochs, n_channels, n_samples = epochs.shape
    n_freqs = len(frequencies)
    
    # Morlet widths: w = w0 * fs / (2 * pi * f), where w0 is central frequency parameter (default 5.0)
    w0 = 5.0
    scales = w0 * fs / (2 * np.pi * frequencies)
    
    # Downsample factor for temporal dimension (T // 4)
    downsample_factor = 4
    n_times_out = n_samples // downsample_factor
    
    scalogram_out = np.zeros((n_epochs, n_channels, n_freqs, n_times_out), dtype=np.float32)
    
    # Loop over epochs and channels
    for e in range(n_epochs):
        for c in range(n_channels):
            sig = epochs[e, c]
            
            # Compute CWT for each scale
            for i, scale in enumerate(scales):
                wavelet_len = min(int(6 * scale), n_samples)
                if wavelet_len % 2 == 0:
                    wavelet_len += 1
                
                t = np.arange(-wavelet_len // 2, wavelet_len // 2 + 1)
                
                # Morlet wavelet formula
                t_scaled = t / scale
                norm = (np.pi ** -0.25) / np.sqrt(scale)
                wavelet_kernel = norm * np.exp(1j * w0 * t_scaled) * np.exp(-0.5 * t_scaled ** 2)
                
                # Convolve signal with wavelet kernel
                coeffs = np.abs(np.convolve(sig, wavelet_kernel, mode='same'))
                
                # Downsample by factor of 4 using mean pooling
                coeffs_downsampled = coeffs[:n_times_out * downsample_factor].reshape(
                    n_times_out, downsample_factor
                ).mean(axis=-1)
                
                scalogram_out[e, c, i] = coeffs_downsampled
                
    return scalogram_out

def batch_cwt(x_raw: torch.Tensor, fs: float = 256.0) -> torch.Tensor:
    """
    Compute CWT on-the-fly for a batch of PyTorch tensors.
    
    Args:
        x_raw: (B, C, T) PyTorch tensor
        fs: sampling rate
    Returns:
        x_scalogram: (B, C, F, T') PyTorch tensor
    """
    device = x_raw.device
    x_raw_np = x_raw.detach().cpu().numpy()
    scalogram_np = cwt_transform(x_raw_np, fs=fs)
    return torch.from_numpy(scalogram_np).to(device)
