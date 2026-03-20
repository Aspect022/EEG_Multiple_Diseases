"""
PhysioNet Apnea-ECG Dataset Loader.

Public dataset for sleep apnea detection using single-lead ECG.
Useful for cross-dataset validation.

Download: https://physionet.org/content/apnea-ecg/1.0.0/
Access: Public (no approval required)

Note: Original dataset has binary labels (apnea/normal per minute).
We aggregate to recording-level severity based on apnea percentage.
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import wfdb
except ImportError:
    raise ImportError("wfdb is required: pip install wfdb")


# =============================================================================
# Constants
# =============================================================================

SEVERITY_LABELS = {
    'Healthy': 0,    # < 5% apnea minutes
    'Mild': 1,       # 5-15% apnea minutes
    'Moderate': 2,   # 15-30% apnea minutes
    'Severe': 3,     # > 30% apnea minutes
}

CLASS_NAMES = ['Healthy', 'Mild', 'Moderate', 'Severe']
NUM_CLASSES = 4


# =============================================================================
# Apnea-ECG Dataset
# =============================================================================

class ApneaECGDataset(Dataset):
    """
    PyTorch Dataset for PhysioNet Apnea-ECG.
    
    Args:
        data_dir: Path to Apnea-ECG dataset root
        split: 'train', 'val', or 'test'
        transform: Optional transform
        sequence_length: Length of ECG sequence (samples)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        sequence_length: int = 3000,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.sequence_length = sequence_length
        self.num_classes = NUM_CLASSES
        
        self.recordings = []
        self.labels = []
        self.apnea_percentages = []
        
        self._load_data()
    
    def _load_data(self):
        """Load Apnea-ECG recordings."""
        # Find all recording files
        record_files = sorted(list(self.data_dir.glob("*.dat")))
        
        if not record_files:
            print(f"  [Apnea-ECG] No .dat files found in {self.data_dir}")
            print(f"  [Apnea-ECG] Expected files: a01.dat, a02.dat, etc.")
            return
        
        # Simple train/val/test split
        n = len(record_files)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        if self.split == 'train':
            files = record_files[:train_end]
        elif self.split == 'val':
            files = record_files[train_end:val_end]
        else:
            files = record_files[val_end:]
        
        for record_file in files:
            try:
                record_name = record_file.stem
                record = wfdb.rdrecord(str(record_file.with_suffix('')), smooth_frames=True)
                annotation = wfdb.rdann(str(record_file.with_suffix('')), 'apnea')
                
                # Extract ECG signal
                ecg = record.p_signal[:, 0]  # Single lead
                
                # Compute apnea percentage from annotations
                apnea_minutes = len([a for a in annotation.sample if a == 1])
                total_minutes = len(annotation.sample)
                apnea_pct = (apnea_minutes / total_minutes) * 100 if total_minutes > 0 else 0
                
                # Convert to severity
                if apnea_pct < 5:
                    label = SEVERITY_LABELS['Healthy']
                elif apnea_pct < 15:
                    label = SEVERITY_LABELS['Mild']
                elif apnea_pct < 30:
                    label = SEVERITY_LABELS['Moderate']
                else:
                    label = SEVERITY_LABELS['Severe']
                
                # Store
                self.recordings.append(ecg)
                self.labels.append(label)
                self.apnea_percentages.append(apnea_pct)
                
            except Exception as e:
                print(f"  [Apnea-ECG] Error loading {record_file}: {e}")
        
        if self.recordings:
            print(f"  [Apnea-ECG] {self.split}: loaded {len(self.labels)} recordings")
            print(f"  [Apnea-ECG] Class distribution: {self.get_class_distribution()}")
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ecg = self.recordings[idx]
        label = int(self.labels[idx])
        
        # Handle variable length
        if len(ecg) > self.sequence_length:
            ecg = ecg[:self.sequence_length]
        else:
            ecg = np.pad(ecg, (0, self.sequence_length - len(ecg)))
        
        # Add channel dimension
        ecg = ecg[np.newaxis, :]  # (1, time)
        
        ecg = np.nan_to_num(ecg, nan=0.0)
        ecg = torch.from_numpy(ecg).float()
        
        if self.transform is not None:
            ecg = self.transform(ecg)
        
        return ecg, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        if len(self.labels) == 0:
            return {}
        unique, counts = np.unique(self.labels, return_counts=True)
        return {CLASS_NAMES[int(u)]: int(c) for u, c in zip(unique, counts)}


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_apnea_ecg_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test DataLoaders for Apnea-ECG."""
    
    train_ds = ApneaECGDataset(data_dir, split='train', transform=transform)
    val_ds = ApneaECGDataset(data_dir, split='val', transform=transform)
    test_ds = ApneaECGDataset(data_dir, split='test', transform=transform)
    
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
    
    print(f"\n  [Apnea-ECG] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    
    return train_loader, val_loader, test_loader
