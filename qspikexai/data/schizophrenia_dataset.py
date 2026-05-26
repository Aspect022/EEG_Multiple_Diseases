import os
import numpy as np
import mne
from .base_dataset import BaseEEGDataset
from ..utils.preprocessing import COMMON_CHANNELS

class SchizophreniaDataset(BaseEEGDataset):
    """
    PhysioNet EEG-Schizophrenia dataset loader.
    Loads EDF files, selects 19 common channels, resamples to 256 Hz.
    Labels: 0 = Healthy Control, 1 = Schizophrenia.
    """
    
    def load_subject_data(self, subject_id: str):
        # subject_id is like 's01', 'h01', etc.
        # Find edf file in data_path
        edf_file = None
        for f in os.listdir(self.data_path):
            if f.lower().startswith(subject_id.lower()) and f.lower().endswith('.edf'):
                edf_file = os.path.join(self.data_path, f)
                break
                
        if edf_file is None or not os.path.exists(edf_file):
            return None, None
            
        # Determine label from filename prefix
        if subject_id.lower().startswith('s'):
            label = 1  # Schizophrenia
        else:
            label = 0  # Healthy Control
            
        try:
            # Read EDF file using MNE
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            
            # Select channels
            channels = COMMON_CHANNELS['schizophrenia']
            
            # Rename channels in MNE raw to standard naming if needed (e.g. FP1 -> Fp1)
            rename_dict = {}
            for ch_name in raw.ch_names:
                for target_ch in channels:
                    if ch_name.lower() == target_ch.lower():
                        rename_dict[ch_name] = target_ch
            raw.rename_channels(rename_dict)
            
            # Reorder and pick channels
            raw.pick_channels(channels, ordered=True)
            
            # Resample to 256 Hz
            raw.resample(self.fs_target, verbose=False)
            
            # Extract signal data as numpy array of shape (19, n_samples)
            data = raw.get_data().astype(np.float32)
            
            return data, label
        except Exception as e:
            print(f"Error loading schizophrenia EDF file {edf_file}: {e}")
            return None, None
