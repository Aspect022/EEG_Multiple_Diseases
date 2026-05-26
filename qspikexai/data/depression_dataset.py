import os
import numpy as np
import scipy.io as sio
import mne
from .base_dataset import BaseEEGDataset
from ..utils.preprocessing import COMMON_CHANNELS

# Index mapping for 10-20 positions on MODMA (128-channel cap) or standard 19 channels
# If the channels are named, we search. If they are raw indices, we can approximate.
MODMA_19_CHANNELS_APPROX = [
    # Approximate indices for standard 19 channels in MODMA 128ch cap if channels are unnamed
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36
]

class DepressionDataset(BaseEEGDataset):
    """
    MODMA and Mumtaz Depression dataset loader.
    Handles loading .mat and .edf files.
    Selects 19 common channels, resamples to 256 Hz.
    Labels: 0 = Healthy Control, 1 = MDD.
    """
    
    def load_subject_data(self, subject_id: str):
        # Find .mat or .edf file in data_path
        file_path = None
        for root, dirs, files in os.walk(self.data_path):
            for f in files:
                if f.lower().startswith(subject_id.lower()) and f.lower().endswith(('.mat', '.edf')):
                    file_path = os.path.join(root, f)
                    break
            if file_path:
                break
                
        if file_path is None or not os.path.exists(file_path):
            # Fallback: scan for any file containing subject_id
            for root, dirs, files in os.walk(self.data_path):
                for f in files:
                    if subject_id.lower() in f.lower() and f.lower().endswith(('.mat', '.edf')):
                        file_path = os.path.join(root, f)
                        break
                if file_path:
                    break
                    
        if file_path is None or not os.path.exists(file_path):
            return None, None
            
        # Determine label
        if 'mdd' in subject_id.lower() or 'dep' in subject_id.lower() or 'mdd' in file_path.lower():
            label = 1  # Major Depressive Disorder
        else:
            label = 0  # Healthy Control
            
        try:
            if file_path.endswith('.edf'):
                # Load EDF file using MNE
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                channels = COMMON_CHANNELS['depression']
                rename_dict = {}
                for ch_name in raw.ch_names:
                    for target_ch in channels:
                        if ch_name.lower() == target_ch.lower():
                            rename_dict[ch_name] = target_ch
                raw.rename_channels(rename_dict)
                raw.pick_channels([ch for ch in channels if ch in raw.ch_names], ordered=True)
                raw.resample(self.fs_target, verbose=False)
                data = raw.get_data().astype(np.float32)
                
                # Pad missing channels with zeros
                if data.shape[0] < len(channels):
                    padded_data = np.zeros((len(channels), data.shape[1]), dtype=np.float32)
                    picked_ch_indices = [channels.index(ch) for ch in raw.ch_names if ch in channels]
                    padded_data[picked_ch_indices] = data
                    data = padded_data
                    
                return data, label
                
            elif file_path.endswith('.mat'):
                # Load MAT file
                mat = sio.loadmat(file_path)
                
                # Look for a 2D array inside the keys that represents the EEG signal
                # For MODMA/Mumtaz, typically key is 'data', 'EEG', or 'eeg'
                data = None
                for key in mat.keys():
                    if not key.startswith('__') and isinstance(mat[key], np.ndarray):
                        # Signal shape is typically (channels, samples) or (samples, channels)
                        arr = mat[key]
                        if arr.ndim == 2:
                            # We expect channels to be the smaller dimension (e.g. 19 or 128)
                            if arr.shape[0] > arr.shape[1]:
                                arr = arr.T
                            data = arr
                            break
                            
                if data is None:
                    return None, None
                    
                # Downsample 128 channels to 19 channels if needed
                n_ch = data.shape[0]
                if n_ch == 128:
                    data = data[MODMA_19_CHANNELS_APPROX]
                elif n_ch > 19:
                    data = data[:19]
                elif n_ch < 19:
                    # Pad to 19 channels
                    padded_data = np.zeros((19, data.shape[1]), dtype=np.float32)
                    padded_data[:n_ch] = data
                    data = padded_data
                    
                # Assume original sampling rate is 250 Hz for MODMA / 256 Hz for Mumtaz
                fs_original = 250.0 if n_ch == 128 else 256.0
                
                # Resample to 256 Hz
                if fs_original != self.fs_target:
                    from scipy.signal import resample
                    num_samples_out = int(data.shape[-1] * self.fs_target / fs_original)
                    resampled_data = []
                    for ch in range(data.shape[0]):
                        resampled_data.append(resample(data[ch], num_samples_out))
                    data = np.stack(resampled_data, axis=0).astype(np.float32)
                    
                return data, label
        except Exception as e:
            print(f"Error loading Depression file {file_path}: {e}")
            return None, None
