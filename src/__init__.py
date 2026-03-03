"""
Advanced ECG Classification Framework.

A comprehensive framework for ECG classification using state-of-the-art
neural network architectures including:
- Spiking Neural Networks (SNN)
- Hybrid Quantum-Classical CNNs
- Swin Transformers

Modules:
- data: Dataset loaders and transforms (PTB-XL, Wavelet Transform)
- models: Neural network architectures
- training: Training utilities and trainers
- utils: Benchmarking and utility functions
"""

from . import data
from . import models
from . import training
from . import utils

__version__ = '0.1.0'
__author__ = 'ECG Classification Team'
