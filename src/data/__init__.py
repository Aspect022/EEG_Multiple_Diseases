"""Data loading and preprocessing utilities for EEG classification.

All imports are lazy — modules are only loaded when explicitly requested.
This prevents hard ImportErrors (e.g. MNE) from blocking unrelated pipelines.
"""

# Nothing is imported here at the module level intentionally.
# Import directly from submodules:
#   from src.data.sleep_edf_dataset import SleepEDFDataset, create_sleep_edf_dataloaders
#   from src.data.boas_dataset import BOASDataset, create_boas_dataloaders
#   from src.data.transforms import create_scalogram_transform

__all__ = [
    'BOASDataset',
    'BIDS_SLEEP_STAGES',
    'CLASS_NAMES',
    'create_boas_dataloaders',
    'CachedScalogramDataset',
    'create_cached_dataloaders',
    'SleepEDFDataset',
    'create_sleep_edf_dataloaders',
    'WaveletTransform',
    'BandpassFilter',
    'Normalize',
    'Compose',
    'create_scalogram_transform',
]
