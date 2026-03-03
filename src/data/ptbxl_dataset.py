"""
PTB-XL Dataset Loader for ECG Classification.

This module provides a PyTorch Dataset implementation for the PTB-XL dataset,
filtered to the top 5 diagnostic superclasses: NORM, MI, STTC, CD, HYP.

Reference: https://physionet.org/content/ptb-xl/1.0.3/
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb


# Top 5 diagnostic superclasses in PTB-XL
DIAGNOSTIC_CLASSES = {
    'NORM': 0,  # Normal ECG
    'MI': 1,    # Myocardial Infarction
    'STTC': 2,  # ST/T Change
    'CD': 3,    # Conduction Disturbance
    'HYP': 4,   # Hypertrophy
}

CLASS_NAMES = list(DIAGNOSTIC_CLASSES.keys())


class PTBXL_Dataset(Dataset):
    """
    PyTorch Dataset for PTB-XL ECG data.
    
    Loads ECG signals and their diagnostic labels, filtering to the
    top 5 diagnostic superclasses.
    
    Args:
        data_dir: Path to PTB-XL dataset root directory.
        sampling_rate: ECG sampling rate (100 or 500 Hz). Default: 100.
        split: Dataset split ('train', 'val', 'test'). Default: 'train'.
        transform: Optional transform to apply to ECG signals.
        target_transform: Optional transform to apply to labels.
        multilabel: If True, returns multi-hot labels. If False, returns
                   single class (first applicable). Default: False.
    
    Attributes:
        data: DataFrame containing filtered samples.
        records: List of record paths.
        labels: Numpy array of labels.
    """
    
    def __init__(
        self,
        data_dir: str,
        sampling_rate: int = 100,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        multilabel: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.sampling_rate = sampling_rate
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.multilabel = multilabel
        self.num_classes = len(DIAGNOSTIC_CLASSES)
        
        # Validate inputs
        assert sampling_rate in [100, 500], "Sampling rate must be 100 or 500 Hz"
        assert split in ['train', 'val', 'test'], "Split must be 'train', 'val', or 'test'"
        
        # Load and filter data
        self.data, self.records, self.labels = self._load_data()
        
    def _load_data(self) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
        """Load PTB-XL metadata and filter to target classes."""
        
        # Load main database file
        database_path = self.data_dir / 'ptbxl_database.csv'
        if not database_path.exists():
            raise FileNotFoundError(
                f"PTB-XL database not found at {database_path}. "
                "Please download from https://physionet.org/content/ptb-xl/"
            )
        
        df = pd.read_csv(database_path, index_col='ecg_id')
        
        # Load SCP statements mapping
        scp_path = self.data_dir / 'scp_statements.csv'
        scp_df = pd.read_csv(scp_path, index_col=0)
        
        # Parse SCP codes from string representation
        df['scp_codes_dict'] = df['scp_codes'].apply(lambda x: eval(x))
        
        # Get diagnostic superclass for each record
        df = self._add_diagnostic_superclass(df, scp_df)
        
        # Filter to records with target diagnostic classes
        df = df[df['diagnostic_superclass'].apply(
            lambda x: any(c in DIAGNOSTIC_CLASSES for c in x) if isinstance(x, list) else False
        )]
        
        # Split data using strat_fold (1-10, recommended split)
        # folds 1-8: train, fold 9: val, fold 10: test
        if self.split == 'train':
            df = df[df['strat_fold'].isin(range(1, 9))]
        elif self.split == 'val':
            df = df[df['strat_fold'] == 9]
        else:  # test
            df = df[df['strat_fold'] == 10]
        
        # Build record paths and labels
        records = []
        labels = []
        
        for idx, row in df.iterrows():
            # Select correct filename based on sampling rate
            if self.sampling_rate == 100:
                filename = row['filename_lr']
            else:
                filename = row['filename_hr']
            
            records.append(str(self.data_dir / filename))
            
            # Create label vector
            if self.multilabel:
                label = np.zeros(self.num_classes, dtype=np.float32)
                for cls in row['diagnostic_superclass']:
                    if cls in DIAGNOSTIC_CLASSES:
                        label[DIAGNOSTIC_CLASSES[cls]] = 1.0
            else:
                # Single-label: take first matching class
                label = -1
                for cls in row['diagnostic_superclass']:
                    if cls in DIAGNOSTIC_CLASSES:
                        label = DIAGNOSTIC_CLASSES[cls]
                        break
            
            labels.append(label)
        
        labels = np.array(labels)
        
        print(f"Loaded {len(records)} records for {self.split} split")
        self._print_class_distribution(labels)
        
        return df.reset_index(), records, labels
    
    def _add_diagnostic_superclass(
        self, df: pd.DataFrame, scp_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Map SCP codes to diagnostic superclasses."""
        
        # Get diagnostic superclass mapping
        diagnostic_mapping = scp_df[scp_df['diagnostic'] == 1]['diagnostic_class'].to_dict()
        
        def get_superclasses(scp_codes: Dict) -> List[str]:
            superclasses = set()
            for code, likelihood in scp_codes.items():
                if code in diagnostic_mapping and likelihood >= 50:
                    superclass = diagnostic_mapping[code]
                    if pd.notna(superclass):
                        superclasses.add(superclass)
            return list(superclasses)
        
        df['diagnostic_superclass'] = df['scp_codes_dict'].apply(get_superclasses)
        return df
    
    def _print_class_distribution(self, labels: np.ndarray) -> None:
        """Print class distribution statistics."""
        print("Class distribution:")
        if self.multilabel:
            for cls_name, cls_idx in DIAGNOSTIC_CLASSES.items():
                count = int(labels[:, cls_idx].sum())
                print(f"  {cls_name}: {count} ({100*count/len(labels):.1f}%)")
        else:
            unique, counts = np.unique(labels, return_counts=True)
            for cls_idx, count in zip(unique, counts):
                if cls_idx >= 0:
                    cls_name = CLASS_NAMES[cls_idx]
                    print(f"  {cls_name}: {count} ({100*count/len(labels):.1f}%)")
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single ECG sample and its label.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (ecg_signal, label) where:
                - ecg_signal: Tensor of shape (12, seq_len) for 12-lead ECG
                - label: Single class index or multi-hot vector
        """
        record_path = self.records[idx]
        label = self.labels[idx]
        
        # Load ECG signal using wfdb
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal  # Shape: (seq_len, 12)
        
        # Transpose to (12, seq_len) for PyTorch convention
        signal = signal.T.astype(np.float32)
        
        # Handle NaN values
        signal = np.nan_to_num(signal, nan=0.0)
        
        # Convert to tensor
        signal = torch.from_numpy(signal)
        
        if self.multilabel:
            label = torch.from_numpy(label)
        else:
            label = torch.tensor(label, dtype=torch.long)
        
        # Apply transforms
        if self.transform is not None:
            signal = self.transform(signal)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return signal, label
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Compute sample weights for balanced sampling.
        
        Returns:
            Tensor of per-sample weights inversely proportional to class frequency.
        """
        if self.multilabel:
            # For multilabel, use inverse frequency of first class
            labels = np.argmax(self.labels, axis=1)
        else:
            labels = self.labels
        
        class_counts = np.bincount(labels[labels >= 0], minlength=self.num_classes)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[labels]
        
        return torch.from_numpy(sample_weights.astype(np.float32))


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    sampling_rate: int = 100,
    transform: Optional[Callable] = None,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train, validation, and test DataLoaders for PTB-XL.
    
    Args:
        data_dir: Path to PTB-XL dataset.
        batch_size: Batch size for DataLoaders.
        sampling_rate: ECG sampling rate (100 or 500 Hz).
        transform: Optional transform to apply.
        num_workers: Number of data loading workers.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler
    
    train_dataset = PTBXL_Dataset(
        data_dir, sampling_rate=sampling_rate, split='train', transform=transform
    )
    val_dataset = PTBXL_Dataset(
        data_dir, sampling_rate=sampling_rate, split='val', transform=transform
    )
    test_dataset = PTBXL_Dataset(
        data_dir, sampling_rate=sampling_rate, split='test', transform=transform
    )
    
    # Use weighted sampling for training to handle class imbalance
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
