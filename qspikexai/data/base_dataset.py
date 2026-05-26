import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Any, Optional, Union
from ..utils.preprocessing import (
    bandpass_filter, notch_filter, resample, 
    reject_bad_channels, zscore_normalize
)
from ..utils.transforms import cwt_transform

class BaseEEGDataset(Dataset):
    """
    Abstract base class for EEG datasets.
    Handles loading, preprocessing, epoching, CWT calculation, and caching.
    """
    def __init__(
        self,
        task: str,
        data_path: str,
        subjects: list,
        window_sec: float,
        overlap_ratio: float = 0.25,
        fs_target: float = 256.0,
        cache_dir: Optional[str] = None,
        preload: bool = True
    ):
        self.task = task
        self.data_path = data_path
        self.subjects = subjects
        self.window_sec = window_sec
        self.overlap_ratio = overlap_ratio
        self.fs_target = fs_target
        self.cache_dir = cache_dir
        self.preload = preload
        
        self.epochs_raw = []       # (n_epochs, n_channels, n_samples)
        self.epochs_scal = []      # (n_epochs, n_channels, n_freqs, n_times)
        self.labels = []           # (n_epochs,)
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            
        self._load_and_process_all()
        
    def _load_and_process_all(self):
        """Iterates over subjects, checks cache, loads/processes raw signal."""
        for subject_id in self.subjects:
            cache_raw_path = None
            cache_scal_path = None
            
            if self.cache_dir:
                cache_raw_path = os.path.join(self.cache_dir, f"{self.task}_{subject_id}_raw.npy")
                cache_scal_path = os.path.join(self.cache_dir, f"{self.task}_{subject_id}_scal.npy")
                cache_lbl_path = os.path.join(self.cache_dir, f"{self.task}_{subject_id}_lbl.npy")
                
                if os.path.exists(cache_raw_path) and os.path.exists(cache_scal_path) and os.path.exists(cache_lbl_path):
                    # Load from cache
                    sub_raw = np.load(cache_raw_path)
                    sub_scal = np.load(cache_scal_path)
                    sub_lbl = np.load(cache_lbl_path)
                    
                    self.epochs_raw.extend(sub_raw)
                    self.epochs_scal.extend(sub_scal)
                    self.labels.extend(sub_lbl)
                    continue
            
            # Load raw signal for subject
            raw_data, subject_labels = self.load_subject_data(subject_id)
            if raw_data is None:
                continue
                
            # Preprocess raw data: shape (channels, samples)
            raw_preprocessed = self.preprocess_subject_data(raw_data)
            
            # Epoch subject data
            sub_raw_epochs, sub_lbl_epochs = self.epoch_subject_data(raw_preprocessed, subject_labels)
            if len(sub_raw_epochs) == 0:
                continue
                
            # Compute CWT scalograms: shape (n_epochs, channels, freqs, times)
            sub_scal_epochs = cwt_transform(sub_raw_epochs, fs=self.fs_target)
            
            # Save to cache if enabled
            if self.cache_dir:
                np.save(cache_raw_path, sub_raw_epochs)
                np.save(cache_scal_path, sub_scal_epochs)
                np.save(cache_lbl_path, sub_lbl_epochs)
                
            self.epochs_raw.extend(sub_raw_epochs)
            self.epochs_scal.extend(sub_scal_epochs)
            self.labels.extend(sub_lbl_epochs)
            
        if len(self.epochs_raw) > 0:
            self.epochs_raw = np.array(self.epochs_raw, dtype=np.float32)
            self.epochs_scal = np.array(self.epochs_scal, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.int64)
        else:
            self.epochs_raw = np.zeros((0, 19, int(self.window_sec * self.fs_target)), dtype=np.float32)
            self.epochs_scal = np.zeros((0, 19, 40, int(self.window_sec * self.fs_target) // 4), dtype=np.float32)
            self.labels = np.zeros((0,), dtype=np.int64)
            
    def load_subject_data(self, subject_id: Any) -> Tuple[Optional[np.ndarray], Optional[Union[int, np.ndarray]]]:
        """Override in subclasses to load raw data for a subject."""
        raise NotImplementedError
        
    def preprocess_subject_data(self, raw_data: np.ndarray) -> np.ndarray:
        """Apply the default EEG preprocessing flow."""
        # 1. Bandpass filter
        data = bandpass_filter(raw_data, low=0.5, high=45.0, fs=self.fs_target)
        # 2. Notch filter
        data = notch_filter(data, freq=50.0, fs=self.fs_target)
        # 3. Artefact rejection
        data = reject_bad_channels(data, threshold_uv=150.0)
        # 4. Z-score normalize
        data = zscore_normalize(data)
        return data.astype(np.float32)
        
    def epoch_subject_data(self, raw_preprocessed: np.ndarray, labels: Union[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Divide continuous signal into epochs."""
        window_size = int(self.window_sec * self.fs_target)
        overlap = int(window_size * self.overlap_ratio)
        step = window_size - overlap
        
        num_samples = raw_preprocessed.shape[-1]
        epochs = []
        epoch_labels = []
        
        start = 0
        idx = 0
        while start + window_size <= num_samples:
            epochs.append(raw_preprocessed[:, start:start + window_size])
            if isinstance(labels, np.ndarray):
                # Map label at middle of window or most common label in window
                mid_idx = start + window_size // 2
                epoch_labels.append(labels[mid_idx] if mid_idx < len(labels) else labels[-1])
            else:
                epoch_labels.append(labels)
            start += step
            
        return np.array(epochs, dtype=np.float32), np.array(epoch_labels, dtype=np.int64)
        
    def __len__(self) -> int:
        return len(self.labels)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_raw = torch.from_numpy(self.epochs_raw[idx])
        x_scal = torch.from_numpy(self.epochs_scal[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x_raw, x_scal, y
