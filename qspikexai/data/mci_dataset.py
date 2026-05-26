import os
import numpy as np
import mne
from .base_dataset import BaseEEGDataset
from ..utils.preprocessing import COMMON_CHANNELS

class MCIDataset(BaseEEGDataset):
    """
    MCI (Mild Cognitive Impairment) dataset loader.
    Handles CAUEEG (EDF) and ds002778 (BIDS) datasets.
    Selects 19 common channels, resamples to 256 Hz.
    Labels: 0 = Healthy Control, 1 = MCI.
    """
    
    def load_subject_data(self, subject_id: str):
        # subject_id is like 'hc01', 'mci01', 'sub-001', etc.
        # Find EDF/FIF file in data_path
        edf_file = None
        for root, dirs, files in os.walk(self.data_path):
            for f in files:
                if f.lower().startswith(subject_id.lower()) and f.lower().endswith(('.edf', '.set', '.fif')):
                    edf_file = os.path.join(root, f)
                    break
            if edf_file:
                break
                
        if edf_file is None or not os.path.exists(edf_file):
            # Fallback: scan for any file containing subject_id
            for root, dirs, files in os.walk(self.data_path):
                for f in files:
                    if subject_id.lower() in f.lower() and f.lower().endswith(('.edf', '.set', '.fif')):
                        edf_file = os.path.join(root, f)
                        break
                if edf_file:
                    break
                    
        if edf_file is None or not os.path.exists(edf_file):
            return None, None
            
        # Determine label from filename or subject_id
        if 'mci' in subject_id.lower() or 'mci' in edf_file.lower():
            label = 1  # MCI
        elif 'ad' in subject_id.lower() or 'ad' in edf_file.lower():
            label = 1  # Map AD/dementia to 1 for binary classification (or 2 for 3-class)
        else:
            label = 0  # Healthy Control
            
        try:
            # Read file using MNE
            if edf_file.endswith('.edf'):
                raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            elif edf_file.endswith('.set'):
                raw = mne.io.read_raw_eeglab(edf_file, preload=True, verbose=False)
            else:
                raw = mne.io.read_raw_fif(edf_file, preload=True, verbose=False)
                
            # Select channels
            channels = COMMON_CHANNELS['mci']
            
            # Map channel names
            rename_dict = {}
            for ch_name in raw.ch_names:
                for target_ch in channels:
                    if ch_name.lower() == target_ch.lower():
                        rename_dict[ch_name] = target_ch
            raw.rename_channels(rename_dict)
            
            # Pick common channels
            raw.pick_channels([ch for ch in channels if ch in raw.ch_names], ordered=True)
            
            # If some common channels are missing, pad with zero channels to preserve 19 channels
            # Resample to 256 Hz
            raw.resample(self.fs_target, verbose=False)
            data = raw.get_data().astype(np.float32)
            
            # Pad missing channels with zeros
            if data.shape[0] < len(channels):
                padded_data = np.zeros((len(channels), data.shape[1]), dtype=np.float32)
                picked_ch_indices = [channels.index(ch) for ch in raw.ch_names if ch in channels]
                padded_data[picked_ch_indices] = data
                data = padded_data
                
            return data, label
        except Exception as e:
            print(f"Error loading MCI file {edf_file}: {e}")
            return None, None
