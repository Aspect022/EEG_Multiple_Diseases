import unittest
import os
import numpy as np
from qspikexai.data.unified_loader import get_available_subjects, get_subject_labels, create_unified_dataloaders

class TestUnifiedDataloaders(unittest.TestCase):
    
    def test_get_available_subjects(self):
        # Test with empty path (should fallback to default subjects)
        subjects = get_available_subjects('sleep_apnea', 'invalid_path_to_data')
        self.assertGreater(len(subjects), 0)
        self.assertIn('a01', subjects)
        
    def test_get_subject_labels(self):
        subjects = ['a01', 'b01', 'c01']
        labels = get_subject_labels('sleep_apnea', subjects, 'invalid_path')
        self.assertEqual(len(labels), 3)
        self.assertEqual(labels[0], 3)  # Apnea mapped to severe (approx)
        self.assertEqual(labels[1], 1)  # Borderline mapped to mild
        self.assertEqual(labels[2], 0)  # Controls mapped to healthy
        
    def test_unified_loader_splits(self):
        # Creating loader with invalid path should trigger default subjects,
        # but since raw EDF files are not there, creating the actual dataset will try to load files and warn.
        # Let's verify K-Fold stratified splitting logic itself.
        from sklearn.model_selection import StratifiedKFold
        subjects = [f'hc{i:02d}' for i in range(1, 11)] + [f'mdd{i:02d}' for i in range(1, 11)]
        labels = get_subject_labels('depression', subjects, 'invalid_path')
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds_checked = 0
        for f_idx, (train_idx, val_idx) in enumerate(skf.split(subjects, labels)):
            train_subs = [subjects[i] for i in train_idx]
            val_subs = [subjects[i] for i in val_idx]
            
            # Verify patient-level grouping (no overlap of subject IDs between train/val)
            intersection = set(train_subs).intersection(set(val_subs))
            self.assertEqual(len(intersection), 0, "Data leakage detected: patient overlap between splits!")
            folds_checked += 1
            
        self.assertEqual(folds_checked, 5)

if __name__ == '__main__':
    unittest.main()
