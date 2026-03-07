"""
Signal Transforms for Deep Learning.

This module provides transforms to convert 1D physiological signals (EEG/ECG)
to 2D representations suitable for vision models (CNNs, Vision Transformers, etc.).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Literal
import warnings


class WaveletTransform:
    """
    Convert 1D signals (EEG/ECG) to 2D scalograms using Continuous Wavelet Transform.
    
    Uses the Morlet wavelet to create time-frequency representations that
    can be fed to vision models like CNNs or Vision Transformers.
    
    Args:
        output_size: Target output size (H, W). Default: (224, 224).
        wavelet: Wavelet type ('morlet', 'ricker'). Default: 'morlet'.
        num_scales: Number of frequency scales. Default: 64.
        sampling_rate: Input signal sampling rate in Hz. Default: 100.
        freq_range: Frequency range (min, max) in Hz. Default: (0.5, 40).
        normalize: Normalization method ('minmax', 'zscore', None). Default: 'minmax'.
        to_rgb: If True, stack scalogram to 3 channels for RGB models. Default: True.
        lead_fusion: How to combine multiple leads ('average', 'stack', 'first').
                    Default: 'average'.
    
    Example:
        >>> transform = WaveletTransform(output_size=(224, 224))
        >>> signal = torch.randn(6, 3000)  # 6-ch EEG, 30s at 100Hz
        >>> scalogram = transform(signal)  # (3, 224, 224)
    """
    
    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),
        wavelet: Literal['morlet', 'ricker'] = 'morlet',
        num_scales: int = 64,
        sampling_rate: int = 100,
        freq_range: Tuple[float, float] = (0.5, 40.0),
        normalize: Optional[Literal['minmax', 'zscore']] = 'minmax',
        to_rgb: bool = True,
        lead_fusion: Literal['average', 'stack', 'first'] = 'average',
    ):
        self.output_size = output_size
        self.wavelet = wavelet
        self.num_scales = num_scales
        self.sampling_rate = sampling_rate
        self.freq_range = freq_range
        self.normalize = normalize
        self.to_rgb = to_rgb
        self.lead_fusion = lead_fusion
        
        # Precompute scales for the wavelet transform
        self.scales = self._compute_scales()
        
    def _compute_scales(self) -> np.ndarray:
        """Compute wavelet scales corresponding to desired frequency range."""
        f_min, f_max = self.freq_range
        
        # For Morlet wavelet, scale ≈ center_freq / frequency
        # Morlet center frequency is approximately 0.8 Hz * scale
        center_freq = 0.8 if self.wavelet == 'morlet' else 0.25
        
        # Compute scale range
        s_min = center_freq * self.sampling_rate / f_max
        s_max = center_freq * self.sampling_rate / f_min
        
        # Logarithmically spaced scales
        scales = np.logspace(np.log10(s_min), np.log10(s_max), self.num_scales)
        
        return scales
    
    def _morlet_wavelet(self, t: np.ndarray, scale: float, omega0: float = 5.0) -> np.ndarray:
        """
        Generate Morlet wavelet at given scale.
        
        Args:
            t: Time array.
            scale: Wavelet scale.
            omega0: Central frequency parameter.
            
        Returns:
            Complex Morlet wavelet.
        """
        t_scaled = t / scale
        norm = (np.pi ** -0.25) / np.sqrt(scale)
        wavelet = norm * np.exp(1j * omega0 * t_scaled) * np.exp(-0.5 * t_scaled ** 2)
        return wavelet
    
    def _ricker_wavelet(self, t: np.ndarray, scale: float) -> np.ndarray:
        """
        Generate Ricker (Mexican hat) wavelet at given scale.
        
        Args:
            t: Time array.
            scale: Wavelet scale.
            
        Returns:
            Real Ricker wavelet.
        """
        t_scaled = t / scale
        norm = 2.0 / (np.sqrt(3 * scale) * (np.pi ** 0.25))
        wavelet = norm * (1 - t_scaled ** 2) * np.exp(-0.5 * t_scaled ** 2)
        return wavelet
    
    def _cwt(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute Continuous Wavelet Transform using convolution.
        
        Args:
            signal: 1D signal array of shape (seq_len,).
            
        Returns:
            Scalogram of shape (num_scales, seq_len).
        """
        seq_len = len(signal)
        scalogram = np.zeros((self.num_scales, seq_len), dtype=np.float32)
        
        for i, scale in enumerate(self.scales):
            # Create wavelet kernel
            # Wavelet support: 6 standard deviations
            wavelet_len = min(int(6 * scale), seq_len)
            if wavelet_len % 2 == 0:
                wavelet_len += 1
            
            t = np.arange(-wavelet_len // 2, wavelet_len // 2 + 1)
            
            if self.wavelet == 'morlet':
                wavelet = self._morlet_wavelet(t, scale)
            else:
                wavelet = self._ricker_wavelet(t, scale)
            
            # Convolve signal with wavelet
            if np.iscomplexobj(wavelet):
                # For complex wavelets, take magnitude
                coeffs = np.abs(np.convolve(signal, wavelet, mode='same'))
            else:
                coeffs = np.convolve(signal, wavelet, mode='same')
            
            scalogram[i] = coeffs
        
        return scalogram
    
    def _resize(self, scalogram: np.ndarray) -> np.ndarray:
        """Resize scalogram to target output size using bilinear interpolation."""
        from scipy.ndimage import zoom
        
        h, w = scalogram.shape
        target_h, target_w = self.output_size
        
        zoom_factors = (target_h / h, target_w / w)
        resized = zoom(scalogram, zoom_factors, order=1)
        
        return resized
    
    def _normalize_scalogram(self, scalogram: np.ndarray) -> np.ndarray:
        """Apply normalization to scalogram."""
        if self.normalize == 'minmax':
            s_min, s_max = scalogram.min(), scalogram.max()
            if s_max - s_min > 1e-8:
                scalogram = (scalogram - s_min) / (s_max - s_min)
            else:
                scalogram = np.zeros_like(scalogram)
        elif self.normalize == 'zscore':
            mean, std = scalogram.mean(), scalogram.std()
            if std > 1e-8:
                scalogram = (scalogram - mean) / std
            else:
                scalogram = np.zeros_like(scalogram)
        
        return scalogram
    
    def __call__(self, ecg: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Transform signal(s) to scalogram(s).
        
        Args:
            ecg: Signal of shape (num_channels, seq_len) or (seq_len,).
            
        Returns:
            Scalogram tensor of shape (C, H, W) where C is 1 or 3.
        """
        # Convert to numpy
        if isinstance(ecg, torch.Tensor):
            ecg = ecg.numpy()
        
        # Ensure 2D: (leads, seq_len)
        if ecg.ndim == 1:
            ecg = ecg[np.newaxis, :]
        
        num_leads, seq_len = ecg.shape
        
        # Compute scalogram for each lead
        scalograms = []
        for lead_idx in range(num_leads):
            signal = ecg[lead_idx]
            
            # Remove DC offset
            signal = signal - np.mean(signal)
            
            # Compute CWT
            scalogram = self._cwt(signal)
            
            # Take absolute value (magnitude)
            scalogram = np.abs(scalogram)
            
            scalograms.append(scalogram)
        
        scalograms = np.stack(scalograms, axis=0)  # (leads, scales, seq_len)
        
        # Fuse leads
        if self.lead_fusion == 'average':
            scalogram = np.mean(scalograms, axis=0)  # (scales, seq_len)
        elif self.lead_fusion == 'first':
            scalogram = scalograms[0]
        elif self.lead_fusion == 'stack':
            # Use first 3 leads for RGB-like representation
            if num_leads >= 3:
                scalogram = scalograms[:3]  # (3, scales, seq_len)
            else:
                # Pad with copies
                scalogram = np.concatenate([
                    scalograms,
                    np.tile(scalograms[:1], (3 - num_leads, 1, 1))
                ], axis=0)
        
        # Normalize
        if self.normalize:
            if scalogram.ndim == 2:
                scalogram = self._normalize_scalogram(scalogram)
            else:
                for i in range(scalogram.shape[0]):
                    scalogram[i] = self._normalize_scalogram(scalogram[i])
        
        # Resize to output size
        if scalogram.ndim == 2:
            scalogram = self._resize(scalogram)
            scalogram = scalogram[np.newaxis, ...]  # Add channel dim
        else:
            resized = []
            for i in range(scalogram.shape[0]):
                resized.append(self._resize(scalogram[i]))
            scalogram = np.stack(resized, axis=0)
        
        # Convert to RGB if needed
        if self.to_rgb and scalogram.shape[0] == 1:
            scalogram = np.tile(scalogram, (3, 1, 1))
        
        # Convert to tensor
        scalogram = torch.from_numpy(scalogram.astype(np.float32))
        
        return scalogram


class BandpassFilter:
    """
    Apply bandpass filtering to physiological signals.
    
    Args:
        low_freq: Low cutoff frequency in Hz. Default: 0.5.
        high_freq: High cutoff frequency in Hz. Default: 40.0.
        sampling_rate: Signal sampling rate in Hz. Default: 100.
        order: Filter order. Default: 4.
    """
    
    def __init__(
        self,
        low_freq: float = 0.5,
        high_freq: float = 40.0,
        sampling_rate: int = 100,
        order: int = 4,
    ):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.sampling_rate = sampling_rate
        self.order = order
        
    def __call__(self, ecg: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Apply bandpass filter to signal."""
        from scipy.signal import butter, filtfilt
        
        # Convert to numpy
        is_tensor = isinstance(ecg, torch.Tensor)
        if is_tensor:
            ecg = ecg.numpy()
        
        # Design filter
        nyquist = self.sampling_rate / 2
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        
        # Ensure valid frequency range
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        b, a = butter(self.order, [low, high], btype='band')
        
        # Apply filter
        if ecg.ndim == 1:
            filtered = filtfilt(b, a, ecg)
        else:
            filtered = np.array([filtfilt(b, a, lead) for lead in ecg])
        
        # Convert back to tensor
        return torch.from_numpy(filtered.astype(np.float32))


class Normalize:
    """
    Normalize physiological signals.
    
    Args:
        method: Normalization method ('zscore', 'minmax', 'robust').
        per_lead: If True, normalize each lead independently. Default: True.
    """
    
    def __init__(
        self,
        method: Literal['zscore', 'minmax', 'robust'] = 'zscore',
        per_lead: bool = True,
    ):
        self.method = method
        self.per_lead = per_lead
        
    def __call__(self, ecg: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Normalize signal."""
        is_tensor = isinstance(ecg, torch.Tensor)
        if is_tensor:
            ecg = ecg.numpy()
        
        if ecg.ndim == 1:
            ecg = ecg[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False
        
        normalized = np.zeros_like(ecg)
        
        for i in range(ecg.shape[0] if self.per_lead else 1):
            if self.per_lead:
                data = ecg[i]
            else:
                data = ecg
            
            if self.method == 'zscore':
                mean, std = data.mean(), data.std()
                result = (data - mean) / (std + 1e-8)
            elif self.method == 'minmax':
                dmin, dmax = data.min(), data.max()
                result = (data - dmin) / (dmax - dmin + 1e-8)
            elif self.method == 'robust':
                median = np.median(data)
                iqr = np.percentile(data, 75) - np.percentile(data, 25)
                result = (data - median) / (iqr + 1e-8)
            
            if self.per_lead:
                normalized[i] = result
            else:
                normalized = result
                break
        
        if squeeze:
            normalized = normalized[0]
        
        return torch.from_numpy(normalized.astype(np.float32))


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
        
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


# Convenience function to create standard transform pipeline
def create_scalogram_transform(
    output_size: Tuple[int, int] = (224, 224),
    sampling_rate: int = 100,
    apply_filter: bool = True,
) -> Compose:
    """
    Create a standard transform pipeline for scalogram generation.
    
    Args:
        output_size: Target image size.
        sampling_rate: Signal sampling rate (Hz).
        apply_filter: Whether to apply bandpass filtering.
        
    Returns:
        Composed transform pipeline.
    """
    transforms = []
    
    if apply_filter:
        transforms.append(BandpassFilter(sampling_rate=sampling_rate))
    
    transforms.append(Normalize(method='zscore'))
    transforms.append(WaveletTransform(
        output_size=output_size,
        sampling_rate=sampling_rate,
        to_rgb=True,
    ))
    
    return Compose(transforms)
