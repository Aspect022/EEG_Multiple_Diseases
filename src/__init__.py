"""
Advanced ECG Classification Framework.

A comprehensive framework for ECG classification using state-of-the-art
neural network architectures including:
- Spiking Neural Networks (SNN) with LIF and QIF neurons
- Hybrid Quantum-Classical CNNs with configurable entanglement/rotations
- Swin Transformers

Modules:
- data: Dataset loaders and transforms (PTB-XL, Wavelet Transform)
- models: Neural network architectures
- training: Training utilities and trainers
- evaluation: Comprehensive metrics (30+)
- utils: Benchmarking and utility functions
"""

from . import data
from . import models
from . import training
from . import utils
from . import evaluation

__version__ = '0.2.0'
__author__ = 'ECG Classification Team'
