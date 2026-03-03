"""
Apnea-ECG Dataset Loader for Sleep Apnea Classification.

This module provides a PyTorch Dataset for the PhysioNet Apnea-ECG Database.
Each overnight ECG recording is segmented into 1-minute epochs and labeled
as Apnea (A) or Normal (N) based on expert annotations.

Reference: https://physionet.org/content/apnea-ecg/1.0.0/
"""

import os
import re
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
import wfdb


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APNEA_CLASSES = {'N': 0, 'A': 1}
CLASS_NAMES = ['Normal', 'Apnea']
SAMPLES_PER_MINUTE = 6000   # 100 Hz × 60 seconds
SAMPLING_RATE = 100          # Hz

# Training set records (a = apnea, b = borderline, c = control)
TRAINING_RECORDS = [
    'a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10',
    'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20',
    'b01', 'b02', 'b03', 'b04', 'b05',
    'c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10',
]

# Held-out test records (x = withheld labels during challenge)
TEST_RECORDS = [
    f'x{i:02d}' for i in range(1, 36)
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ApneaECGDataset(Dataset):
    """
    PyTorch Dataset for the PhysioNet Apnea-ECG Database.

    Segments overnight single-channel ECG recordings into 1-minute epochs
    and assigns binary labels (Normal=0, Apnea=1) from `.apn` annotations.

    Args:
        data_dir:   Path to `apnea-ecg-database-1.0.0/` folder.
        fold:       Cross-validation fold index (0–4).  Ignored if ``split='test'``.
        n_folds:    Total number of CV folds. Default: 5.
        split:      ``'train'``, ``'val'``, or ``'test'``.
        transform:  Optional callable applied to each 1-min ECG segment
                    *after* it has been converted to a Tensor of shape
                    ``(1, 6000)``.
        random_state: Seed for reproducible fold assignment.

    Attributes:
        segments:    List of ``(record_name, minute_index)`` tuples.
        labels:      numpy int array of labels aligned with ``segments``.
        patient_ids: List of patient-record identifiers for leakage checking.
    """

    def __init__(
        self,
        data_dir: str,
        fold: int = 0,
        n_folds: int = 5,
        split: str = 'train',
        transform: Optional[Callable] = None,
        random_state: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.fold = fold
        self.n_folds = n_folds
        self.split = split
        self.transform = transform
        self.random_state = random_state
        self.num_classes = 2

        assert split in ('train', 'val', 'test'), \
            f"split must be 'train', 'val', or 'test', got '{split}'"
        assert 0 <= fold < n_folds, \
            f"fold must be in [0, {n_folds}), got {fold}"

        # Build index of (record, minute_idx) → label
        self.segments, self.labels, self.patient_ids = self._build_index()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def _build_index(self) -> Tuple[List[Tuple[str, int]], np.ndarray, List[str]]:
        """Build a flat list of segments with labels."""

        if self.split == 'test':
            records = TEST_RECORDS
        else:
            records = TRAINING_RECORDS

        all_segments: List[Tuple[str, int]] = []
        all_labels: List[int] = []
        all_patient_ids: List[str] = []

        for rec_name in records:
            rec_path = self.data_dir / rec_name

            # Check that signal header and data exist
            if not (self.data_dir / f"{rec_name}.hea").exists():
                continue
            if not (self.data_dir / f"{rec_name}.dat").exists():
                continue

            # Load annotations
            apn_path = self.data_dir / f"{rec_name}.apn"
            if not apn_path.exists():
                # x-records may not have .apn during challenge; skip if absent
                continue

            try:
                ann = wfdb.rdann(str(rec_path), 'apn')
                # Load header to get signal length
                hdr = wfdb.rdheader(str(rec_path))
            except Exception:
                # If either file is corrupt or rdann/rdheader fail, skip this record
                continue

            # Each annotation sample index corresponds to the *start* of a minute
            # Labels are stored in ann.symbol ('A' or 'N')
            minute_labels = {}
            for sample_idx, symbol in zip(ann.sample, ann.symbol):
                minute_idx = sample_idx // SAMPLES_PER_MINUTE
                if symbol in APNEA_CLASSES:
                    minute_labels[minute_idx] = APNEA_CLASSES[symbol]

            total_minutes = hdr.sig_len // SAMPLES_PER_MINUTE

            for minute_idx in range(total_minutes):
                if minute_idx in minute_labels:
                    all_segments.append((rec_name, minute_idx))
                    all_labels.append(minute_labels[minute_idx])
                    all_patient_ids.append(rec_name)

        all_labels = np.array(all_labels, dtype=np.int64)

        # For test split, return everything directly
        if self.split == 'test':
            return all_segments, all_labels, all_patient_ids

        # For train/val, perform patient-level stratified group k-fold
        return self._apply_fold_split(all_segments, all_labels, all_patient_ids)

    def _apply_fold_split(
        self,
        segments: List[Tuple[str, int]],
        labels: np.ndarray,
        patient_ids: List[str],
    ) -> Tuple[List[Tuple[str, int]], np.ndarray, List[str]]:
        """Apply stratified group k-fold splitting at the patient level."""

        groups = np.array(patient_ids)
        sgkf = StratifiedGroupKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(segments, labels, groups)):
            if fold_idx == self.fold:
                if self.split == 'train':
                    idx = train_idx
                else:  # val
                    idx = val_idx
                break

        selected_segments = [segments[i] for i in idx]
        selected_labels = labels[idx]
        selected_patients = [patient_ids[i] for i in idx]

        return selected_segments, selected_labels, selected_patients

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            signal: Tensor of shape ``(1, 6000)`` or transformed shape.
            label:  Integer 0 (Normal) or 1 (Apnea).
        """
        rec_name, minute_idx = self.segments[idx]
        label = int(self.labels[idx])

        # Read the specific 1-minute segment
        start_sample = minute_idx * SAMPLES_PER_MINUTE
        end_sample = start_sample + SAMPLES_PER_MINUTE

        rec_path = str(self.data_dir / rec_name)
        record = wfdb.rdrecord(rec_path, sampfrom=start_sample, sampto=end_sample)
        signal = record.p_signal.T.astype(np.float32)  # (1, 6000)

        # Handle NaN
        signal = np.nan_to_num(signal, nan=0.0)

        signal = torch.from_numpy(signal)

        if self.transform is not None:
            signal = self.transform(signal)

        return signal, label

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def get_class_distribution(self) -> Dict[str, int]:
        """Return count of each class."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return {CLASS_NAMES[int(u)]: int(c) for u, c in zip(unique, counts)}

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for loss function."""
        counts = np.bincount(self.labels, minlength=2)
        weights = len(self.labels) / (2.0 * counts + 1e-6)
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        counts = np.bincount(self.labels, minlength=2)
        class_weights = 1.0 / (counts.astype(np.float32) + 1e-6)
        sample_weights = class_weights[self.labels]
        return torch.from_numpy(sample_weights)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def create_apnea_dataloaders(
    data_dir: str,
    fold: int = 0,
    n_folds: int = 5,
    batch_size: int = 32,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders for a single CV fold.

    Args:
        data_dir:     Path to ``apnea-ecg-database-1.0.0/``.
        fold:         Fold index (0-based).
        n_folds:      Total folds.
        batch_size:   Batch size.
        num_workers:  Workers for data loading.
        transform:    Optional transform applied to each segment.
        random_state: Seed for fold reproducibility.

    Returns:
        ``(train_loader, val_loader)``
    """
    from torch.utils.data import WeightedRandomSampler

    train_ds = ApneaECGDataset(
        data_dir, fold=fold, n_folds=n_folds, split='train',
        transform=transform, random_state=random_state,
    )
    val_ds = ApneaECGDataset(
        data_dir, fold=fold, n_folds=n_folds, split='val',
        transform=transform, random_state=random_state,
    )

    # Balanced sampling for training
    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    print(f"Fold {fold}/{n_folds} | "
          f"Train: {len(train_ds)} segments ({train_ds.get_class_distribution()}) | "
          f"Val: {len(val_ds)} segments ({val_ds.get_class_distribution()})")

    return train_loader, val_loader


def create_test_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
) -> DataLoader:
    """Create a DataLoader for the held-out x-records test set."""
    test_ds = ApneaECGDataset(
        data_dir, fold=0, split='test', transform=transform,
    )
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
