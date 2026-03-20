"""
SHHS (Sleep Heart Health Study) Dataset Loader.

SHHS is a large cohort study of sleep-disordered breathing with PSG recordings
and AHI (Apnea-Hypopnea Index) annotations for sleep apnea severity classification.

Download: https://sleepdata.org/datasets/shhs
Access: Requires NSRR data use agreement (1-2 weeks approval)

Severity Classification (4-class):
    - Healthy:    AHI < 5
    - Mild:       5 <= AHI < 15
    - Moderate:   15 <= AHI < 30
    - Severe:     AHI >= 30
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    import mne
    mne.set_log_level('ERROR')
except ImportError:
    raise ImportError("MNE is required: pip install mne")


# =============================================================================
# Constants
# =============================================================================

SEVERITY_LABELS = {
    'Healthy': 0,    # AHI < 5
    'Mild': 1,       # 5 <= AHI < 15
    'Moderate': 2,   # 15 <= AHI < 30
    'Severe': 3,     # AHI >= 30
}

CLASS_NAMES = ['Healthy', 'Mild', 'Moderate', 'Severe']
NUM_CLASSES = 4

# AHI thresholds
AHI_THRESHOLDS = {
    'Healthy': (0, 5),
    'Mild': (5, 15),
    'Moderate': (15, 30),
    'Severe': (30, float('inf')),
}

# Default PSG channels
DEFAULT_CHANNELS = ['C3-A2', 'C4-A1', 'EOG-L', 'EOG-R', 'EMG', 'Resp']
EPOCH_DURATION = 30  # seconds


# =============================================================================
# SHHS Dataset
# =============================================================================

class SHHSDataset(Dataset):
    """
    PyTorch Dataset for SHHS sleep apnea severity classification.
    
    Args:
        data_dir: Path to SHHS dataset root
        split: 'train', 'val', or 'test'
        transform: Optional transform applied to each sample
        channels: PSG channels to use
        target_sfreq: Target sampling frequency (Hz)
        aggregation: How to aggregate epochs for recording-level prediction
                    'mean' - Average epoch features
                    'attention' - Attention-weighted aggregation
                    'lstm' - Use LSTM for temporal modeling
        max_subjects: Limit subjects for debugging
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        channels: List[str] = DEFAULT_CHANNELS,
        target_sfreq: float = 125.0,
        aggregation: str = 'mean',
        max_subjects: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.channels = channels
        self.target_sfreq = target_sfreq
        self.aggregation = aggregation
        self.num_classes = NUM_CLASSES
        
        # Load data
        self.recordings = []
        self.labels = []
        self.ahi_values = []
        
        self._load_data(max_subjects)
    
    def _load_data(self, max_subjects: Optional[int] = None):
        """Load SHHS recordings and AHI labels."""
        # TODO: Implement SHHS-specific loading logic
        # This is a skeleton - actual implementation depends on SHHS file structure
        
        print(f"  [SHHS] Loading {self.split} split from {self.data_dir}")
        print(f"  [SHHS] ⚠️  SKELETON IMPLEMENTATION - Replace with actual SHHS loader")
        
        # Placeholder data for testing
        # Replace with actual SHHS loading code
        self.recordings = np.random.randn(100, len(self.channels), int(EPOCH_DURATION * self.target_sfreq)).astype(np.float32)
        self.labels = np.random.randint(0, NUM_CLASSES, size=100)
        self.ahi_values = np.random.uniform(0, 50, size=100)
        
        print(f"  [SHHS] Loaded {len(self.labels)} recordings")
        print(f"  [SHHS] Class distribution: {self.get_class_distribution()}")
    
    def _ahi_to_severity(self, ahi: float) -> int:
        """Convert AHI value to severity class."""
        if ahi < 5:
            return SEVERITY_LABELS['Healthy']
        elif ahi < 15:
            return SEVERITY_LABELS['Mild']
        elif ahi < 30:
            return SEVERITY_LABELS['Moderate']
        else:
            return SEVERITY_LABELS['Severe']
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Returns:
            signal: (channels, time) for single epoch
                    OR (num_epochs, channels, time) for sequence
            label: severity class (0-3)
        """
        signal = self.recordings[idx]
        label = int(self.labels[idx])
        
        # Handle NaN values
        signal = np.nan_to_num(signal, nan=0.0)
        signal = torch.from_numpy(signal).float()
        
        if self.transform is not None:
            signal = self.transform(signal)
        
        return signal, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of severity classes."""
        if len(self.labels) == 0:
            return {}
        unique, counts = np.unique(self.labels, return_counts=True)
        return {CLASS_NAMES[int(u)]: int(c) for u, c in zip(unique, counts)}
    
    def get_ahi_statistics(self) -> Dict[str, float]:
        """Get AHI statistics."""
        return {
            'mean': float(np.mean(self.ahi_values)),
            'std': float(np.std(self.ahi_values)),
            'min': float(np.min(self.ahi_values)),
            'max': float(np.max(self.ahi_values)),
            'median': float(np.median(self.ahi_values)),
        }


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_shhs_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    target_sfreq: float = 125.0,
    aggregation: str = 'mean',
    max_subjects: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test DataLoaders for SHHS.
    
    Args:
        data_dir: Path to SHHS dataset root
        batch_size: Batch size
        num_workers: DataLoader workers
        transform: Transform to apply to each sample
        target_sfreq: Target sampling frequency
        aggregation: Aggregation method for sequence modeling
        max_subjects: Limit subjects for debugging
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = SHHSDataset(
        data_dir, split='train', transform=transform,
        target_sfreq=target_sfreq, aggregation=aggregation,
        max_subjects=max_subjects,
    )
    val_ds = SHHSDataset(
        data_dir, split='val', transform=transform,
        target_sfreq=target_sfreq, aggregation=aggregation,
        max_subjects=max_subjects,
    )
    test_ds = SHHSDataset(
        data_dir, split='test', transform=transform,
        target_sfreq=target_sfreq, aggregation=aggregation,
        max_subjects=max_subjects,
    )
    
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    
    print(f"\n  [SHHS] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    
    return train_loader, val_loader, test_loader
