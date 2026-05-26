import numpy as np
import scipy.signal as signal
from typing import Dict, List, Union

FS_TARGET = 256  # target sampling rate in Hz

COMMON_CHANNELS = {
    'sleep_apnea':     ['ECG'],
    'schizophrenia':   ['Fp1','Fp2','F7','F3','Fz','F4',
                        'F8','T3','C3','Cz','C4','T4',
                        'T5','P3','Pz','P4','T6','O1','O2'],
    'mci':             ['Fp1','Fp2','F7','F3','Fz','F4',
                        'F8','T3','C3','Cz','C4','T4',
                        'T5','P3','Pz','P4','T6','O1','O2'],
    'depression':      ['Fp1','Fp2','F7','F3','Fz','F4',
                        'F8','T3','C3','Cz','C4','T4',
                        'T5','P3','Pz','P4','T6','O1','O2'],
}

WINDOW_SEC = {
    'sleep_apnea':    60,
    'schizophrenia':  10,
    'mci':            8,
    'depression':     8,
}

def bandpass_filter(data: np.ndarray, low: float = 0.5, high: float = 45.0, fs: float = 256.0, order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = signal.butter(order, [low_norm, high_norm], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)

def notch_filter(data: np.ndarray, freq: float = 50.0, fs: float = 256.0) -> np.ndarray:
    """Apply zero-phase notch filter to remove line noise."""
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = signal.iirnotch(w0, Q=30.0)
    return signal.filtfilt(b, a, data, axis=-1)

def resample(data: np.ndarray, fs_original: float, fs_target: float = 256.0) -> np.ndarray:
    """Resample signal to target frequency."""
    if fs_original == fs_target:
        return data
    num_samples = int(data.shape[-1] * fs_target / fs_original)
    return signal.resample(data, num_samples, axis=-1)

def reject_bad_channels(data: np.ndarray, threshold_uv: float = 150.0) -> np.ndarray:
    """Zero out channels whose amplitude exceeds the threshold (in microvolts)."""
    clean_data = data.copy()
    for ch in range(data.shape[0]):
        if np.max(np.abs(data[ch])) > threshold_uv:
            clean_data[ch] = 0.0
    return clean_data

def zscore_normalize(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """Z-score normalization along the specified axis."""
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    return (data - mean) / (std + 1e-8)

def epoch_signal(data: np.ndarray, window_size: int, overlap: int) -> List[np.ndarray]:
    """Epoch signal into fixed-length windows."""
    step = window_size - overlap
    num_samples = data.shape[-1]
    epochs = []
    start = 0
    while start + window_size <= num_samples:
        epochs.append(data[:, start:start + window_size])
        start += step
    return epochs
