import os
import numpy as np
import wfdb
from .base_dataset import BaseEEGDataset

class SleepApneaDataset(BaseEEGDataset):
    """
    PhysioNet Apnea-ECG dataset loader.
    Segments ECG signals into 1-minute epochs (15,360 samples at 256 Hz).
    Labels are mapped to 4 severity classes based on the subject's overall AHI.
    """
    
    def load_subject_data(self, subject_id: str):
        # subject_id is like 'a01', 'c02', etc.
        rec_path = os.path.join(self.data_path, subject_id)
        
        # Check files
        if not os.path.exists(f"{rec_path}.hea") or not os.path.exists(f"{rec_path}.dat"):
            return None, None
            
        # Load annotations to calculate subject AHI
        apn_path = f"{rec_path}.apn"
        if not os.path.exists(apn_path):
            return None, None
            
        try:
            ann = wfdb.rdann(rec_path, 'apn')
            hdr = wfdb.rdheader(rec_path)
        except Exception as e:
            print(f"Error reading wfdb records for {subject_id}: {e}")
            return None, None
            
        # Count apnea minutes
        apnea_count = sum(1 for sym in ann.symbol if sym == 'A')
        total_minutes = len(ann.symbol)
        
        if total_minutes == 0:
            return None, None
            
        # Calculate AHI: apnea events per hour
        ahi = 60.0 * apnea_count / total_minutes
        
        # Map AHI to 4-class severity: Healthy (<5), Mild (5–15), Moderate (15–30), Severe (>30)
        if ahi < 5.0:
            label = 0  # Healthy
        elif ahi < 15.0:
            label = 1  # Mild
        elif ahi < 30.0:
            label = 2  # Moderate
        else:
            label = 3  # Severe
            
        # Read the p_signal (1 channel ECG)
        record = wfdb.rdrecord(rec_path)
        signal = record.p_signal.T.astype(np.float32)  # (n_channels, n_samples)
        
        # Select channel 0 (typically only 1 channel ECG is present)
        signal = signal[0:1] # Keep it 2D: (1, n_samples)
        
        # Handle NaN
        signal = np.nan_to_num(signal, nan=0.0)
        
        # Resample from original fs (100 Hz) to target fs (256 Hz)
        fs_original = record.fs
        signal_resampled = []
        for ch in range(signal.shape[0]):
            # Use scipy.signal.resample from the imported utils or directly
            from scipy.signal import resample
            num_samples_out = int(signal.shape[-1] * self.fs_target / fs_original)
            signal_resampled.append(resample(signal[ch], num_samples_out))
            
        signal_resampled = np.stack(signal_resampled, axis=0).astype(np.float32)
        
        return signal_resampled, label
