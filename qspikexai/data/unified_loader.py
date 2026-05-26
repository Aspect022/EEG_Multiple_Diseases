import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .sleep_apnea_dataset import SleepApneaDataset
from .schizophrenia_dataset import SchizophreniaDataset
from .mci_dataset import MCIDataset
from .depression_dataset import DepressionDataset

# Predefined subject lists for datasets where filenames are fixed
DEFAULT_SUBJECTS = {
    'sleep_apnea': [
        'a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10',
        'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20',
        'b01', 'b02', 'b03', 'b04', 'b05',
        'c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10'
    ],
    'schizophrenia': [
        'h01', 'h02', 'h03', 'h04', 'h05', 'h06', 'h07', 'h08', 'h09', 'h10',
        'h11', 'h12', 'h13', 'h14',
        's01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10',
        's11', 's12', 's13', 's14'
    ],
    'mci': [f'hc{i:02d}' for i in range(1, 56)] + [f'mci{i:02d}' for i in range(1, 56)],
    'depression': [f'hc{i:02d}' for i in range(1, 30)] + [f'mdd{i:02d}' for i in range(1, 25)]
}

def get_available_subjects(task: str, data_path: str) -> list:
    """Scan the data directory to find available subject IDs."""
    if not os.path.exists(data_path):
        return DEFAULT_SUBJECTS[task]
        
    files = os.listdir(data_path)
    if len(files) == 0:
        return DEFAULT_SUBJECTS[task]
        
    subjects = set()
    if task == 'sleep_apnea':
        # Find all .hea files that match a\d+, b\d+, c\d+
        for f in files:
            if f.endswith('.hea'):
                name = os.path.splitext(f)[0]
                if name.startswith(('a', 'b', 'c', 'x')):
                    subjects.add(name)
    elif task == 'schizophrenia':
        for f in files:
            if f.endswith('.edf'):
                # Subject prefix is typically like s01, h01
                name = os.path.splitext(f)[0]
                subjects.add(name[:3])
    elif task == 'mci':
        for root, dirs, filenames in os.walk(data_path):
            for f in filenames:
                if f.endswith(('.edf', '.set', '.fif')):
                    name = os.path.splitext(f)[0]
                    # Try to extract subject code
                    subjects.add(name)
    elif task == 'depression':
        for root, dirs, filenames in os.walk(data_path):
            for f in filenames:
                if f.endswith(('.mat', '.edf')):
                    name = os.path.splitext(f)[0]
                    subjects.add(name)
                    
    subjects_list = sorted(list(subjects))
    return subjects_list if len(subjects_list) > 0 else DEFAULT_SUBJECTS[task]

def get_subject_labels(task: str, subjects: list, data_path: str) -> np.ndarray:
    """Determine labels for a list of subjects for stratified splitting."""
    labels = []
    for sub in subjects:
        if task == 'sleep_apnea':
            # Borderline and apnea subjects are labeled 1/2/3, controls 0
            if sub.startswith('c'):
                labels.append(0)
            elif sub.startswith('b'):
                labels.append(1)  # Borderline maps to mild
            else:
                labels.append(3)  # Apnea mapped to severe (approx)
        elif task == 'schizophrenia':
            labels.append(1 if sub.lower().startswith('s') else 0)
        elif task == 'mci':
            labels.append(1 if 'mci' in sub.lower() or 'ad' in sub.lower() else 0)
        elif task == 'depression':
            labels.append(1 if 'mdd' in sub.lower() or 'dep' in sub.lower() else 0)
    return np.array(labels)

def get_dataset_class(task: str):
    """Return appropriate dataset class for the task."""
    if task == 'sleep_apnea':
        return SleepApneaDataset
    elif task == 'schizophrenia':
        return SchizophreniaDataset
    elif task == 'mci':
        return MCIDataset
    elif task == 'depression':
        return DepressionDataset
    else:
        raise ValueError(f"Unknown task: {task}")

def create_unified_dataloaders(
    task: str,
    data_dir: str,
    fold: int = 0,
    n_folds: int = 5,
    batch_size: int = 32,
    num_workers: int = 0,
    window_sec: float = None,
    overlap_ratio: float = 0.25,
    cache_dir: str = None,
    seed: int = 42
):
    """
    Unified dataloader factory. Performs subject-level stratified splitting.
    """
    if window_sec is None:
        from ..utils.preprocessing import WINDOW_SEC as PREPROCESSING_WINDOW_SEC
        window_sec = PREPROCESSING_WINDOW_SEC[task]
        
    # Get subjects
    subjects = get_available_subjects(task, data_dir)
    labels = get_subject_labels(task, subjects, data_dir)
    
    # Stratified K-Fold at Subject level
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    train_subjects = []
    val_subjects = []
    
    # Get subjects for the specified fold
    for f_idx, (train_idx, val_idx) in enumerate(skf.split(subjects, labels)):
        if f_idx == fold:
            train_subjects = [subjects[i] for i in train_idx]
            val_subjects = [subjects[i] for i in val_idx]
            break
            
    dataset_class = get_dataset_class(task)
    
    train_dataset = dataset_class(
        task=task,
        data_path=data_dir,
        subjects=train_subjects,
        window_sec=window_sec,
        overlap_ratio=overlap_ratio,
        cache_dir=cache_dir
    )
    
    val_dataset = dataset_class(
        task=task,
        data_path=data_dir,
        subjects=val_subjects,
        window_sec=window_sec,
        overlap_ratio=overlap_ratio,
        cache_dir=cache_dir
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
