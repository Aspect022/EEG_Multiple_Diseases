"""
Sleep-EDF Expanded dataset utilities.

Provides subject-level train/val/test splits for PhysioNet Sleep-EDF Expanded
and supports:
  - raw 1D EEG epochs for temporal models
  - 2D scalograms via external transforms
  - paired raw + scalogram samples for multimodal fusion
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")


SLEEP_STAGES = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": -1,
    "Movement time": -1,
}

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
EPOCH_DURATION = 30
DEFAULT_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
TARGET_NUM_CHANNELS = 6
MIN_RECOMMENDED_RECORD_PAIRS = 10
STORAGE_DTYPE = np.float16


def verify_sleep_edf_dataset(data_dir: str, min_pairs: int = MIN_RECOMMENDED_RECORD_PAIRS) -> bool:
    """Return True when the directory has enough matched Sleep-EDF record pairs."""
    root = Path(data_dir)
    if not root.exists():
        return False
    return len(_discover_record_pairs(root)) >= min_pairs


def _extract_subject_id(record_name: str) -> str:
    """
    Extract a stable subject identifier from Sleep-EDF file names.

    Example:
      SC4001E0 -> SC4001
      ST7022J0 -> ST7022
    """
    match = re.match(r"([A-Z]{2}\d{4})", record_name)
    return match.group(1) if match else record_name[:6]


def _discover_record_pairs(data_dir: Path) -> List[Dict[str, str]]:
    """Find PSG/Hypnogram file pairs."""
    pairs: List[Dict[str, str]] = []
    for psg_file in sorted(data_dir.rglob("*PSG.edf")):
        record_prefix = psg_file.name[:7]
        hyp_files = sorted(psg_file.parent.glob(f"{record_prefix}*Hypnogram.edf"))
        if not hyp_files:
            hyp_files = sorted(data_dir.rglob(f"{record_prefix}*Hypnogram.edf"))
        if not hyp_files:
            continue

        record_name = psg_file.stem.replace("-PSG", "")
        pairs.append(
            {
                "record_name": record_name,
                "subject_id": _extract_subject_id(record_name),
                "psg": str(psg_file),
                "hypnogram": str(hyp_files[0]),
            }
        )
    return pairs


def _split_records_by_subject(
    records: Sequence[Dict[str, str]],
    split: str,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Create deterministic subject-level splits."""
    subjects = sorted({r["subject_id"] for r in records})
    rng = np.random.default_rng(seed)
    subjects = list(rng.permutation(subjects))

    n_subjects = len(subjects)
    if n_subjects < 3:
        train_subjects = set(subjects)
        val_subjects = set()
        test_subjects = set()
    else:
        n_val = max(1, int(0.15 * n_subjects))
        n_test = max(1, int(0.15 * n_subjects))
        n_train = max(1, n_subjects - n_val - n_test)

        # Ensure we always keep at least one subject for each split once there
        # are enough subjects to do so.
        if n_train + n_val + n_test > n_subjects:
            overflow = (n_train + n_val + n_test) - n_subjects
            n_train = max(1, n_train - overflow)

        train_subjects = set(subjects[:n_train])
        val_subjects = set(subjects[n_train:n_train + n_val])
        test_subjects = set(subjects[n_train + n_val:n_train + n_val + n_test])

    if split == "train":
        selected = train_subjects
    elif split == "val":
        selected = val_subjects
    elif split == "test":
        selected = test_subjects
    else:
        raise ValueError(f"Unknown split: {split}")

    return [r for r in records if r["subject_id"] in selected]


def _pad_or_trim_channels(epoch: np.ndarray, target_channels: int) -> np.ndarray:
    """Normalize channel count so 1D models can keep a fixed input shape."""
    current_channels, time_steps = epoch.shape
    if current_channels == target_channels:
        return epoch
    if current_channels > target_channels:
        return epoch[:target_channels]

    padded = np.zeros((target_channels, time_steps), dtype=epoch.dtype)
    padded[:current_channels] = epoch
    return padded


class SleepEDFDataset(Dataset):
    """
    Subject-level Sleep-EDF dataset with raw epoch extraction.

    Returns 30-second epochs with shape `(target_num_channels, target_sfreq*30)`.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        channels: Optional[List[str]] = None,
        target_sfreq: float = 100.0,
        target_num_channels: int = TARGET_NUM_CHANNELS,
        max_records: Optional[int] = None,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.channels = channels or list(DEFAULT_CHANNELS)
        self.target_sfreq = target_sfreq
        self.target_num_channels = target_num_channels
        self.num_classes = len(CLASS_NAMES)
        self.seed = seed

        records = _discover_record_pairs(self.data_dir)
        records = _split_records_by_subject(records, split=split, seed=seed)
        if max_records is not None:
            records = records[:max_records]

        self.records = records
        self.epochs_data: np.ndarray
        self.labels: np.ndarray
        self._load_records()

    def _cache_path(self) -> Path:
        """Deterministic cache filename based on split + settings."""
        max_tag = f"_max{self.max_records}" if self.max_records is not None else ""
        tag = f"{self.split}_sfreq{int(self.target_sfreq)}_ch{self.target_num_channels}_seed{self.seed}{max_tag}"
        cache_dir = self.data_dir / ".cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"sleep_edf_{tag}.npz"

    def _load_records(self) -> None:
        cache_path = self._cache_path()
        data_file = cache_path.with_name(f"{cache_path.stem}_data.npy")
        label_file = cache_path.with_name(f"{cache_path.stem}_labels.npy")

        samples_per_epoch = int(EPOCH_DURATION * self.target_sfreq)

        # ── Fast path: load from memmap ──────────────────────────────────
        if data_file.exists() and label_file.exists():
            print(f"  [Sleep-EDF] {self.split}: loading from memmap cache {data_file.name} ...", flush=True)
            self.labels = np.load(label_file, allow_pickle=False)
            shape = (len(self.labels), self.target_num_channels, samples_per_epoch)
            self.epochs_data = np.memmap(data_file, dtype=STORAGE_DTYPE, mode='r', shape=shape)
            print(
                f"  [Sleep-EDF] {self.split}: {len(self.labels)} epochs "
                f"({self.get_class_distribution()})", flush=True,
            )
            return

        # ── Slow path: read EDF files and stream to memmap ─────────────
        print(f"  [Sleep-EDF] {self.split}: building cache from {len(self.records)} records ...", flush=True)
        
        # Pass 1: Count total epochs to pre-allocate memmap (takes seconds)
        total_epochs = 0
        valid_events = []
        for i, record in enumerate(self.records):
            try:
                ann = mne.read_annotations(record["hypnogram"])
                record_events = []
                for a in ann:
                    desc = str(a["description"]).strip()
                    lbl = SLEEP_STAGES.get(desc, -1)
                    if lbl >= 0:
                        full_eps = int(float(a["duration"]) // EPOCH_DURATION)
                        for e_idx in range(full_eps):
                            onset = float(a["onset"]) + e_idx * EPOCH_DURATION
                            record_events.append((onset, lbl))
                total_epochs += len(record_events)
                valid_events.append(record_events)
            except Exception:
                valid_events.append([])

        if total_epochs == 0:
            self.epochs_data = np.empty((0, self.target_num_channels, samples_per_epoch), dtype=STORAGE_DTYPE)
            self.labels = np.empty((0,), dtype=np.int64)
            print(f"  [Sleep-EDF] {self.split}: WARNING - no epochs loaded", flush=True)
            return

        shape = (total_epochs, self.target_num_channels, samples_per_epoch)
        print(f"  [Sleep-EDF] {self.split}: allocating {shape} memmap on disk (zero RAM overhead)...", flush=True)
        
        # Allocate Memmap
        self.epochs_data = np.memmap(data_file, dtype=STORAGE_DTYPE, mode='w+', shape=shape)
        self.labels = np.zeros(total_epochs, dtype=np.int64)
        
        # Pass 2: Extract data and stream directly to disk
        idx = 0
        loaded_records = 0
        
        for i, record in enumerate(self.records):
            events = valid_events[i]
            if not events: continue
            
            try:
                raw = mne.io.read_raw_edf(record["psg"], preload=True, verbose=False)
                available = [ch for ch in self.channels if ch in raw.ch_names]
                if not available: continue
                
                raw.pick_channels(available)
                if raw.info["sfreq"] != self.target_sfreq:
                    raw.resample(self.target_sfreq)
                    
                sfreq = raw.info["sfreq"]
                signal = raw.get_data().astype(np.float32)
                
                for (onset, label) in events:
                    start_sample = int(onset * sfreq)
                    end_sample = start_sample + samples_per_epoch
                    if end_sample > signal.shape[1]: continue
                        
                    epoch = signal[:, start_sample:end_sample]
                    if epoch.shape[1] != samples_per_epoch: continue
                        
                    epoch = _pad_or_trim_channels(epoch, self.target_num_channels)
                    
                    self.epochs_data[idx] = epoch.astype(STORAGE_DTYPE)
                    self.labels[idx] = label
                    idx += 1
                
                self.epochs_data.flush()  # Flush directly to disk to keep RAM empty
                loaded_records += 1
                
                if (i + 1) % 10 == 0 or (i + 1) == len(self.records):
                    print(f"  [Sleep-EDF] {self.split}: {i+1}/{len(self.records)} records extracted ({idx} epochs streamed to disk)", flush=True)
                    
            except Exception as exc:
                print(f"  [Sleep-EDF] Failed loading {record['record_name']}: {exc}", flush=True)

        self.epochs_data.flush()
        
        # Save exact labels array (in case some were dropped due to out-of-bounds)
        np.save(label_file, self.labels[:idx])
        
        # Reload memmap in read-only mode to the EXACT size of extracted epochs
        del self.epochs_data
        shape = (idx, self.target_num_channels, samples_per_epoch)
        self.epochs_data = np.memmap(data_file, dtype=STORAGE_DTYPE, mode='r', shape=shape)
        self.labels = self.labels[:idx]
        
        print(f"  [Sleep-EDF] {self.split}: {len(self.labels)} epochs from {loaded_records} records — cached to {data_file.name}", flush=True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        signal = np.nan_to_num(self.epochs_data[idx].astype(np.float32, copy=False), nan=0.0)
        signal_tensor = torch.from_numpy(signal).float()
        label = int(self.labels[idx])

        if self.transform is not None:
            signal_tensor = self.transform(signal_tensor)

        return signal_tensor, label

    def get_class_distribution(self) -> Dict[str, int]:
        if len(self.labels) == 0:
            return {}
        unique, counts = np.unique(self.labels, return_counts=True)
        return {CLASS_NAMES[int(u)]: int(c) for u, c in zip(unique, counts)}


class SleepEDFMultiModalDataset(Dataset):
    """Paired raw-signal + scalogram Sleep-EDF dataset."""

    def __init__(self, base_dataset: SleepEDFDataset, scalogram_transform: Callable):
        self.base_dataset = base_dataset
        self.scalogram_transform = scalogram_transform
        self.labels = base_dataset.labels
        self.num_classes = base_dataset.num_classes

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        signal = np.nan_to_num(self.base_dataset.epochs_data[idx].astype(np.float32, copy=False), nan=0.0)
        raw_signal = torch.from_numpy(signal).float()
        scalogram = self.scalogram_transform(raw_signal.clone())
        label = int(self.base_dataset.labels[idx])
        return raw_signal, scalogram, label


def create_sleep_edf_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    target_sfreq: float = 100.0,
    max_records: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create raw or transformed Sleep-EDF dataloaders."""
    train_ds = SleepEDFDataset(
        data_dir, split="train", transform=transform,
        target_sfreq=target_sfreq, max_records=max_records,
    )
    val_ds = SleepEDFDataset(
        data_dir, split="val", transform=transform,
        target_sfreq=target_sfreq, max_records=max_records,
    )
    test_ds = SleepEDFDataset(
        data_dir, split="test", transform=transform,
        target_sfreq=target_sfreq, max_records=max_records,
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    print(f"\n  Train: {len(train_ds)} epochs | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


def create_sleep_edf_multimodal_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    scalogram_transform: Optional[Callable] = None,
    target_sfreq: float = 100.0,
    max_records: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create paired raw + scalogram dataloaders."""
    if scalogram_transform is None:
        raise ValueError("scalogram_transform is required for multimodal Sleep-EDF loading")

    train_base = SleepEDFDataset(
        data_dir, split="train", transform=None,
        target_sfreq=target_sfreq, max_records=max_records,
    )
    val_base = SleepEDFDataset(
        data_dir, split="val", transform=None,
        target_sfreq=target_sfreq, max_records=max_records,
    )
    test_base = SleepEDFDataset(
        data_dir, split="test", transform=None,
        target_sfreq=target_sfreq, max_records=max_records,
    )

    train_ds = SleepEDFMultiModalDataset(train_base, scalogram_transform)
    val_ds = SleepEDFMultiModalDataset(val_base, scalogram_transform)
    test_ds = SleepEDFMultiModalDataset(test_base, scalogram_transform)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    print(f"\n  Train: {len(train_ds)} paired epochs | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
