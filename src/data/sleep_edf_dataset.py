import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import mne

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SLEEP_STAGES = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,  # Merge N3 and N4 into N3 (Stage 3)
    'Sleep stage R': 4
}
CLASS_NAMES = ['W', 'N1', 'N2', 'N3', 'REM']
EPOCH_DURATION = 30  # seconds

class SleepEDFDataset(Dataset):
    """
    PyTorch Dataset for PhysioNet Sleep-EDF Expanded.
    Extracts 30-second epochs and labels from paired PSG and Hypnogram EDF files.
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        channels: List[str] = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal'],
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.channels = channels
        self.num_classes = 5
        
        self.epochs_data = []
        self.labels = []
        
        self._load_data()

    def _load_data(self):
        # Find all PSG files
        psg_files = sorted(list(self.data_dir.glob("*PSG.edf")))
        
        for psg_file in psg_files:
            # Find matching hypnogram file
            base_name = psg_file.name[:7] # e.g. SC4001E
            hypno_files = list(self.data_dir.glob(f"{base_name}*Hypnogram.edf"))
            if not hypno_files:
                continue
            hypno_file = hypno_files[0]
            
            # Load raw PSG
            raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
            
            # Keep only required channels
            raw.pick_channels([ch for ch in self.channels if ch in raw.ch_names])
            
            # Identify sampling rate
            sfreq = raw.info['sfreq']
            
            # Load annotations
            annot = mne.read_annotations(hypno_file)
            raw.set_annotations(annot, emit_warning=False)
            
            # Extract events from annotations
            events, event_id = mne.events_from_annotations(
                raw, event_id=SLEEP_STAGES, chunk_duration=EPOCH_DURATION, verbose=False)
            
            # Create epochs
            epochs = mne.Epochs(raw, events, event_id, tmin=0., tmax=EPOCH_DURATION - 1.0/sfreq,
                                baseline=None, preload=True, verbose=False)
            
            epochs_data = epochs.get_data() # (n_epochs, n_channels, n_times)
            events_labels = epochs.events[:, 2] # (n_epochs,)
            
            # Optionally resample to 100 Hz if needed by transformations later, 
            # but usually transform handles filtering or resampling if we pass it out
            # Let's keep raw shape here, and let user transforms handle it
            
            self.epochs_data.append(epochs_data)
            self.labels.append(events_labels)
            
        if self.epochs_data:
            self.epochs_data = np.concatenate(self.epochs_data, axis=0) # (Total_epochs, C, T)
            self.labels = np.concatenate(self.labels, axis=0)           # (Total_epochs,)
            
            # Mock split for now: first 80% train, 10% val, 10% test
            total = len(self.labels)
            if self.split == 'train':
                self.epochs_data = self.epochs_data[:int(0.8 * total)]
                self.labels = self.labels[:int(0.8 * total)]
            elif self.split == 'val':
                self.epochs_data = self.epochs_data[int(0.8 * total):int(0.9 * total)]
                self.labels = self.labels[int(0.8 * total):int(0.9 * total)]
            elif self.split == 'test':
                self.epochs_data = self.epochs_data[int(0.9 * total):]
                self.labels = self.labels[int(0.9 * total):]
        else:
            self.epochs_data = np.array([])
            self.labels = np.array([])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        signal = self.epochs_data[idx] # (C, T)
        label = int(self.labels[idx])
        
        # Avoid NaNs
        signal = np.nan_to_num(signal, nan=0.0)
        
        signal = torch.from_numpy(signal).float()
        
        if self.transform is not None:
            signal = self.transform(signal)
            
        return signal, label

    def get_class_distribution(self) -> Dict[str, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return {CLASS_NAMES[int(u)]: int(c) for u, c in zip(unique, counts)}

def create_sleep_edf_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader]:

    train_ds = SleepEDFDataset(data_dir, split='train', transform=transform)
    val_ds = SleepEDFDataset(data_dir, split='val', transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    print(f"Train: {len(train_ds)} segments ({train_ds.get_class_distribution()}) | "
          f"Val: {len(val_ds)} segments ({val_ds.get_class_distribution()})")

    return train_loader, val_loader
