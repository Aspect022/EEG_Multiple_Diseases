"""
Severity label definitions for sleep apnea classification.
"""

# Severity class mappings
SEVERITY_LABELS = {
    'Healthy': 0,    # AHI < 5
    'Mild': 1,       # 5 <= AHI < 15
    'Moderate': 2,   # 15 <= AHI < 30
    'Severe': 3,     # AHI >= 30
}

# Reverse mapping
SEVERITY_NAMES = {v: k for k, v in SEVERITY_LABELS.items()}

# Class names in order
CLASS_NAMES = ['Healthy', 'Mild', 'Moderate', 'Severe']

# Number of classes
NUM_CLASSES = 4

# AHI thresholds
AHI_THRESHOLDS = {
    'Healthy': (0, 5),
    'Mild': (5, 15),
    'Moderate': (15, 30),
    'Severe': (30, float('inf')),
}

# Clinical descriptions
CLINICAL_DESCRIPTIONS = {
    'Healthy': 'No sleep apnea (AHI < 5)',
    'Mild': 'Mild sleep apnea (5 ≤ AHI < 15)',
    'Moderate': 'Moderate sleep apnea (15 ≤ AHI < 30)',
    'Severe': 'Severe sleep apnea (AHI ≥ 30)',
}
