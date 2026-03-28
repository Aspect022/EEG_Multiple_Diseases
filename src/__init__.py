"""
Advanced EEG Classification Framework.

Sub-packages are imported lazily to avoid hard dependency crashes
(e.g. MNE, PennyLane) when only a subset of the framework is used.

Modules:
- data: Dataset loaders and transforms (BOAS, Sleep-EDF, CWT)
- models: Neural network architectures
- training: Training utilities and trainers
- evaluation: Comprehensive metrics (30+)
- utils: Benchmarking and utility functions
"""

__version__ = '0.3.0'
__author__ = 'EEG Classification Team'

# Submodules are NOT imported here — import them explicitly as needed:
#   from src.training.research_trainer import FoldTrainer, ResearchConfig
#   from src.data.sleep_edf_dataset import create_sleep_edf_dataloaders
#   from src.models.fusion.conditional_routing_fusion import create_conditional_routing
