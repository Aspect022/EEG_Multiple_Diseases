"""
BOAS (Bitbrain Open Access Sleep) Dataset Loader — OpenNeuro ds005555.

BIDS-format EEG dataset: 128 nights of PSG recordings with expert-consensus
5-class sleep staging annotations (W / N1 / N2 / N3 / REM).

PSG EEG channels: F3, F4, C3, C4, O1, O2 (6 channels).
Epoch duration: 30 seconds.

Download:
    pip install awscli
    aws s3 sync --no-sign-request s3://openneuro.org/ds005555 data/ds005555/
"""

import os
import re
import warnings

# Suppress noisy MNE warnings about mixed channel filters and undefined physical ranges
warnings.filterwarnings('ignore', category=RuntimeWarning)
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import mne
    mne.set_log_level('ERROR')
except ImportError:
    raise ImportError("MNE is required for BOAS loader. Install: pip install mne")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLEEP_STAGES = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,   # Merge N4 into N3
    'Sleep stage R': 4,
    'Sleep stage ?': -1,  # Unknown — skip
    'Movement time': -1,  # Movement — skip
}

# BIDS-format stage mappings (from events.tsv)
BIDS_SLEEP_STAGES = {
    'Wake': 0, 'W': 0,
    'N1': 1, 'Stage 1': 1, 'NREM1': 1,
    'N2': 2, 'Stage 2': 2, 'NREM2': 2,
    'N3': 3, 'Stage 3': 3, 'NREM3': 3, 'Stage 4': 3,
    'REM': 4, 'R': 4,
}

CLASS_NAMES = ['W', 'N1', 'N2', 'N3', 'REM']
NUM_CLASSES = 5
EPOCH_DURATION = 30  # seconds

# Target PSG EEG channels
PSG_EEG_CHANNELS = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
# Alternative channel naming in BOAS
PSG_EEG_ALTERNATIVES = {
    'F3': ['PSG_F3', 'EEG F3-A2', 'EEG F3', 'F3-A2', 'F3'],
    'F4': ['PSG_F4', 'EEG F4-A1', 'EEG F4', 'F4-A1', 'F4'],
    'C3': ['PSG_C3', 'EEG C3-A2', 'EEG C3', 'C3-A2', 'C3'],
    'C4': ['PSG_C4', 'EEG C4-A1', 'EEG C4', 'C4-A1', 'C4'],
    'O1': ['PSG_O1', 'EEG O1-A2', 'EEG O1', 'O1-A2', 'O1'],
    'O2': ['PSG_O2', 'EEG O2-A1', 'EEG O2', 'O2-A1', 'O2'],
}


# ---------------------------------------------------------------------------
# BOAS Dataset
# ---------------------------------------------------------------------------

class BOASDataset(Dataset):
    """
    PyTorch Dataset for OpenNeuro BOAS (ds005555).

    Reads BIDS-format PSG EEG recordings, extracts 30-second epochs,
    and assigns 5-class sleep stage labels.

    Args:
        data_dir: Path to ds005555 root (contains sub-001/, sub-002/, etc.)
        split: 'train', 'val', or 'test' (subject-level split).
        transform: Optional transform applied to each epoch signal.
        channels: EEG channels to use. Default: 6 PSG EEG channels.
        target_sfreq: Target sampling frequency for resampling. Default: 100 Hz.
        max_subjects: Limit subjects loaded (for debugging). None = all.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        channels: List[str] = PSG_EEG_CHANNELS,
        target_sfreq: float = 100.0,
        max_subjects: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.channels = channels
        self.target_sfreq = target_sfreq
        self.num_classes = NUM_CLASSES

        # Discover subjects
        all_subjects = self._discover_subjects()

        # Subject-level split (no data leakage)
        train_subs, val_subs, test_subs = self._split_subjects(all_subjects)

        if split == 'train':
            subjects = train_subs
        elif split == 'val':
            subjects = val_subs
        elif split == 'test':
            subjects = test_subs
        else:
            raise ValueError(f"Unknown split: {split}")

        if max_subjects is not None:
            subjects = subjects[:max_subjects]

        # Load all epochs
        self.epochs_data = []
        self.labels = []
        self._load_subjects(subjects)

    def _discover_subjects(self) -> List[str]:
        """Find all subject directories in BIDS layout."""
        subjects = sorted([
            d.name for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.startswith('sub-')
        ])
        return subjects

    def _split_subjects(self, subjects: List[str]):
        """Split subjects into train/val/test (70/15/15)."""
        np.random.seed(42)
        n = len(subjects)
        indices = np.random.permutation(n)

        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        return (
            [subjects[i] for i in train_idx],
            [subjects[i] for i in val_idx],
            [subjects[i] for i in test_idx],
        )

    def _find_eeg_file(self, subject_dir: Path) -> Optional[Path]:
        """Find the PSG EEG file in a BIDS subject directory."""
        # BIDS structure: sub-XXX/ses-XXX/eeg/*_psg_eeg.edf
        # or: sub-XXX/eeg/*_eeg.edf
        patterns = [
            '**/eeg/*psg*eeg*.edf',
            '**/eeg/*_eeg.edf',
            '**/*psg*.edf',
            '**/*.edf',
        ]
        for pattern in patterns:
            files = sorted(subject_dir.glob(pattern))
            if files:
                # Prefer PSG over headband
                psg_files = [f for f in files if 'headband' not in f.name.lower()]
                return psg_files[0] if psg_files else files[0]
        return None

    def _find_events_file(self, eeg_file: Path) -> Optional[Path]:
        """Find the events TSV file matching an EEG file."""
        # BIDS: replace _eeg.edf with _events.tsv
        events_name = re.sub(r'_eeg\.edf$', '_events.tsv', eeg_file.name, flags=re.IGNORECASE)
        events_file = eeg_file.parent / events_name
        if events_file.exists():
            return events_file

        # Try broader search
        for f in eeg_file.parent.glob('*events*.tsv'):
            return f
        return None

    def _parse_annotations_from_raw(self, raw) -> List[Tuple[float, float, int]]:
        """Extract sleep staging from MNE annotations embedded in the EDF."""
        annotations = raw.annotations
        events = []
        for ann in annotations:
            onset = ann['onset']
            duration = ann['duration']
            desc = ann['description'].strip()

            # Try BIDS-style stage names
            if desc in BIDS_SLEEP_STAGES:
                label = BIDS_SLEEP_STAGES[desc]
                events.append((onset, duration, label))
            # Try Sleep-EDF style
            elif desc in SLEEP_STAGES:
                label = SLEEP_STAGES[desc]
                if label >= 0:
                    events.append((onset, duration, label))

        return events

    def _parse_events_tsv(self, events_file: Path) -> List[Tuple[float, float, int]]:
        """Parse BIDS events.tsv for sleep staging.

        BOAS format: columns = onset, duration, begsample, endsample, offset, stage_hum, stage_ai
        Stage values are integers: 0=W, 1=N1, 2=N2, 3=N3, 4=REM, negative=unknown/skip
        """
        df = pd.read_csv(events_file, sep='\t')
        events = []

        # Identify onset and duration columns
        onset_col = 'onset' if 'onset' in df.columns else df.columns[0]
        dur_col = 'duration' if 'duration' in df.columns else df.columns[1]

        # Find stage column — prefer human expert label, then AI, then generic
        stage_col = None
        for col in ['stage_hum', 'stage_ai', 'trial_type', 'value', 'stage',
                     'sleep_stage', 'description']:
            if col in df.columns:
                stage_col = col
                break
        if stage_col is None:
            # Last resort: use the last column
            stage_col = df.columns[-1] if len(df.columns) > 2 else None

        if stage_col is None:
            return events

        for _, row in df.iterrows():
            onset = float(row[onset_col])
            duration = float(row[dur_col]) if pd.notna(row.get(dur_col)) else EPOCH_DURATION

            stage_val = row[stage_col]

            # Handle numeric stages (BOAS format: 0=W, 1=N1, 2=N2, 3=N3, 4=REM)
            try:
                label = int(float(stage_val))
            except (ValueError, TypeError):
                # Text label — try dictionary lookups
                stage_str = str(stage_val).strip()
                label = BIDS_SLEEP_STAGES.get(stage_str, -1)
                if label < 0:
                    label = SLEEP_STAGES.get(stage_str, -1)

            # Only keep valid stages (0-4), skip negatives and >4
            if 0 <= label <= 4:
                events.append((onset, duration, label))

        return events

    def _match_channels(self, raw) -> List[str]:
        """Find matching channels in the raw recording."""
        available = raw.ch_names
        matched = []

        for target in self.channels:
            alternatives = PSG_EEG_ALTERNATIVES.get(target, [target])
            found = False
            for alt in alternatives:
                if alt in available:
                    matched.append(alt)
                    found = True
                    break
            if not found:
                # Try case-insensitive partial match
                for ch in available:
                    if target.lower() in ch.lower():
                        matched.append(ch)
                        found = True
                        break

        return matched

    def _load_subjects(self, subjects: List[str]):
        """Load all EEG epochs from the given subjects."""
        loaded_count = 0

        for sub in subjects:
            sub_dir = self.data_dir / sub
            eeg_file = self._find_eeg_file(sub_dir)

            if eeg_file is None:
                print(f"  [BOAS] Skipping {sub}: no EEG file found")
                continue

            try:
                # Load raw EEG (suppress MNE channel filter warnings)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    raw = mne.io.read_raw_edf(str(eeg_file), preload=True, verbose=False)

                # Match channels
                matched_chs = self._match_channels(raw)
                if len(matched_chs) < 2:
                    print(f"  [BOAS] Skipping {sub}: only {len(matched_chs)} channels matched")
                    continue

                raw.pick_channels(matched_chs)

                # Resample if needed
                if raw.info['sfreq'] != self.target_sfreq:
                    raw.resample(self.target_sfreq)

                sfreq = raw.info['sfreq']
                samples_per_epoch = int(EPOCH_DURATION * sfreq)
                data = raw.get_data()  # (n_channels, n_samples)

                # Get sleep staging events
                events_file = self._find_events_file(eeg_file)
                if events_file is not None:
                    stage_events = self._parse_events_tsv(events_file)
                else:
                    stage_events = self._parse_annotations_from_raw(raw)

                # Debug: show exactly what happened (first 3 subjects only)
                if loaded_count < 3 or not stage_events:
                    print(f"  [DEBUG] {sub}: edf={eeg_file.name}, "
                          f"events={'FOUND: ' + events_file.name if events_file else 'NONE'}, "
                          f"parsed={len(stage_events)} events")

                if not stage_events:
                    print(f"  [BOAS] Skipping {sub}: no sleep staging annotations found")
                    continue

                # Extract epochs
                for onset, duration, label in stage_events:
                    start_sample = int(onset * sfreq)
                    end_sample = start_sample + samples_per_epoch

                    if end_sample > data.shape[1]:
                        continue

                    epoch = data[:, start_sample:end_sample]

                    # Verify shape
                    if epoch.shape[1] == samples_per_epoch:
                        self.epochs_data.append(epoch)
                        self.labels.append(label)

                loaded_count += 1

            except Exception as e:
                print(f"  [BOAS] Error loading {sub}: {e}")
                continue

        if self.epochs_data:
            self.epochs_data = np.array(self.epochs_data, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.int64)
            print(f"  [BOAS] {self.split}: loaded {len(self.labels)} epochs "
                  f"from {loaded_count} subjects ({self.get_class_distribution()})")
        else:
            self.epochs_data = np.empty((0, len(self.channels), int(EPOCH_DURATION * self.target_sfreq)), dtype=np.float32)
            self.labels = np.empty(0, dtype=np.int64)
            print(f"  [BOAS] {self.split}: WARNING — no epochs loaded!")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        signal = self.epochs_data[idx]  # (C, T)
        label = int(self.labels[idx])

        signal = np.nan_to_num(signal, nan=0.0)
        signal = torch.from_numpy(signal).float()

        if self.transform is not None:
            signal = self.transform(signal)

        return signal, label

    def get_class_distribution(self) -> Dict[str, int]:
        if len(self.labels) == 0:
            return {}
        unique, counts = np.unique(self.labels, return_counts=True)
        return {CLASS_NAMES[int(u)]: int(c) for u, c in zip(unique, counts)}


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

def create_boas_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    target_sfreq: float = 100.0,
    max_subjects: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test DataLoaders for BOAS ds005555.

    Args:
        data_dir: Path to ds005555 root.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        transform: CWT transform to convert (C, T) → (3, 224, 224).
        target_sfreq: Target sampling frequency.
        max_subjects: Limit subjects (for debugging).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = BOASDataset(data_dir, split='train', transform=transform,
                           target_sfreq=target_sfreq, max_subjects=max_subjects)
    val_ds = BOASDataset(data_dir, split='val', transform=transform,
                         target_sfreq=target_sfreq, max_subjects=max_subjects)
    test_ds = BOASDataset(data_dir, split='test', transform=transform,
                          target_sfreq=target_sfreq, max_subjects=max_subjects)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)

    print(f"\n  Train: {len(train_ds)} epochs | Val: {len(val_ds)} | Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader
