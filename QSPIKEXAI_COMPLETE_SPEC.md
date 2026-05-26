# QSpikeXAI-Net — Complete Implementation Specification
> **For AI Coder Use**  
> Quantum-Enhanced Spiking Neural Network with Multi-Level XAI  
> Unified EEG Classification: Sleep Apnea · Schizophrenia · MCI · Depression  
> Target: Q1 Journal (IEEE TNSRE / Neural Networks)  
> Platform: A100 GPU Server · Python 3.10+ · PyTorch 2.x

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure (New Clean Pipeline)](#2-repository-structure)
3. [Datasets — All Four Tasks](#3-datasets)
4. [Preprocessing Specification](#4-preprocessing-specification)
5. [Architecture: QSpikeXAI-Net](#5-architecture-qspikexai-net)
6. [Component-Level Code Specification](#6-component-level-code-specification)
7. [Training Protocol](#7-training-protocol)
8. [Baseline Models to Implement](#8-baseline-models)
9. [XAI Module Specification](#9-xai-module-specification)
10. [Experiment Runner](#10-experiment-runner)
11. [Validation Checklist](#11-validation-checklist)
12. [Expected Results & Paper Tables](#12-expected-results)

---

## 1. Project Overview

### What This Is

A **fresh, clean codebase** that replaces the existing fragmented pipeline. Do **not** extend `pipeline.py` or `unified_pipeline.py` from the old repo. Start from scratch using this spec.

### The Four Clinical Tasks

| ID | Disorder | Task Type | Labels | Key EEG Biomarker |
|----|----------|-----------|--------|-------------------|
| `sleep_apnea` | Obstructive Sleep Apnea | 4-class severity | Healthy / Mild / Moderate / Severe | Arousal micro-events, delta-beta coupling, SpO2-correlated power |
| `schizophrenia` | Schizophrenia | Binary (extend to 3-class) | HC / SCZ (/ FEP) | Reduced gamma coherence, disrupted alpha, altered P300 |
| `mci` | Mild Cognitive Impairment | Binary or 3-class | HC / MCI (/ AD) | Theta/delta increase, alpha slowing, reduced complexity |
| `depression` | Major Depressive Disorder | Binary | HC / MDD | Frontal alpha asymmetry (FAA), elevated theta, reduced beta |

### Three Core Novel Claims (for the paper)

1. **Quantum-SNN dual-stream fusion** with confidence-gated conditional routing — first application to multi-disorder EEG
2. **Quantum gate attribution XAI** — gradient-based attribution into VQC rotation parameters, not just input channels
3. **Unified framework** across four neurological/psychiatric conditions with a shared backbone and disorder-specific heads

---

## 2. Repository Structure

**Build exactly this structure. No extras.**

```
qspikexai/
├── README.md
├── requirements.txt
├── DATASETS.md                    ← download instructions (copy from Section 3)
│
├── data/                          ← raw downloaded data goes here (git-ignored)
│   ├── apnea-ecg/                 ← PhysioNet Apnea-ECG (already downloaded)
│   ├── shhs/                      ← SHHS subset (optional, large)
│   ├── eeg-schizophrenia/         ← PhysioNet 14+14 subjects
│   ├── schizophrenia-84/          ← OpenNeuro ds003774 (84 subjects)
│   ├── caueeg/                    ← CAUEEG MCI dataset (GitHub)
│   ├── mci-erp/                   ← OpenNeuro ds002778
│   └── modma/                     ← MODMA depression dataset
│
├── qspikexai/                     ← main Python package
│   ├── __init__.py
│   │
│   ├── data/                      ← dataset loaders
│   │   ├── __init__.py
│   │   ├── base_dataset.py        ← abstract base class
│   │   ├── sleep_apnea_dataset.py
│   │   ├── schizophrenia_dataset.py
│   │   ├── mci_dataset.py
│   │   ├── depression_dataset.py
│   │   └── unified_loader.py      ← single entry point
│   │
│   ├── models/                    ← all model code
│   │   ├── __init__.py
│   │   ├── baselines/
│   │   │   ├── __init__.py
│   │   │   ├── eegnet.py
│   │   │   ├── eeg_tcnet.py
│   │   │   ├── resnet1d.py
│   │   │   └── vit1d.py
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── snn_stream.py      ← SNN-1D-Attention stream
│   │   │   ├── vqc_layer.py       ← Variational Quantum Circuit
│   │   │   ├── quantum_stream.py  ← VQC-ViT scalogram stream
│   │   │   ├── router.py          ← confidence-gated conditional router
│   │   │   └── task_heads.py      ← disorder-specific classification heads
│   │   └── qspikexai_net.py       ← main proposed model
│   │
│   ├── xai/                       ← explainability
│   │   ├── __init__.py
│   │   ├── signal_xai.py          ← channel/temporal/band attribution
│   │   └── quantum_xai.py         ← quantum gate attribution (novel)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             ← main training loop
│   │   └── metrics.py             ← all evaluation metrics
│   │
│   └── utils/
│       ├── __init__.py
│       ├── preprocessing.py       ← signal filtering, epoching, normalisation
│       ├── transforms.py          ← CWT scalogram computation
│       └── experiment_logger.py   ← CSV result logging
│
├── experiments/
│   ├── run_baselines.py           ← runs all Tier 1 baselines
│   ├── run_ablations.py           ← runs Tier 2 ablation models
│   ├── run_proposed.py            ← runs full QSpikeXAI-Net
│   └── run_xai.py                 ← computes and saves all XAI outputs
│
├── results/
│   └── canonical_results.csv      ← SINGLE source of truth for all results
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_results_analysis.ipynb
│   └── 03_xai_visualisation.ipynb
│
└── tests/
    ├── test_dataloaders.py         ← CPU-only forward pass tests
    ├── test_models.py              ← model shape tests (no data needed)
    └── test_xai.py                 ← XAI output shape tests
```

---

## 3. Datasets

### 3.1 Sleep Apnea

#### Primary: PhysioNet Apnea-ECG (ALREADY DOWNLOADED)
- **Location in old repo:** `data/apnea-ecg-database-1.0.0/`  
- **Copy command:** `cp -r <old_repo>/data/apnea-ecg-database-1.0.0 data/apnea-ecg/`
- **Subjects:** 70 overnight recordings
- **Signal:** ECG + apnea annotation per minute
- **Labels:** Per-recording AHI → Healthy (<5), Mild (5–15), Moderate (15–30), Severe (>30)
- **URL:** https://physionet.org/content/apnea-ecg/1.0.0/
- **Formats:** `.dat`, `.hea`, `.apn` files

#### Secondary: SHHS Subset (optional, large — use only if compute budget allows)
```bash
wget -r -N -c -np https://physionet.org/files/shhs/1.0.0/
```
- **Warning:** Full dataset is ~200GB. Download subset: records 1–200 for prototyping.

---

### 3.2 Schizophrenia

#### Primary: PhysioNet EEG-Schizophrenia (Olejarczyk & Jernajczyk 2017)
```bash
wget -r -N -c -np https://physionet.org/files/eeg-schizophrenia/1.0.0/
# or using physionet tool:
pip install wfdb
python -c "import wfdb; wfdb.dl_database('eeg-schizophrenia', 'data/eeg-schizophrenia/')"
```
- **Subjects:** 14 SCZ patients + 14 Healthy Controls
- **Channels:** 19 (10-20 system), referenced to linked mastoids
- **Sample rate:** 250 Hz
- **Duration:** ~15 min resting state, eyes closed
- **Format:** EDF files, one per subject
- **Labels:** Binary — `0` = HC, `1` = SCZ
- **Citation:** Olejarczyk & Jernajczyk, *Clin Neurophysiol*, 2017

#### Secondary: OpenNeuro ds003774 (84 subjects — use for validation)
```bash
pip install openneuro-py
openneuro download --dataset ds003774 --target data/schizophrenia-84/
```
- **Subjects:** 42 SCZ + 42 HC
- **Channels:** 32 (BrainProducts actiCAP)
- **Sample rate:** 1000 Hz (downsample to 256 Hz in preprocessing)
- **Conditions:** Eyes-open and eyes-closed resting state
- **Format:** BIDS `.set` / `.fdt` (EEGLAB format)
- **URL:** https://openneuro.org/datasets/ds003774

---

### 3.3 Mild Cognitive Impairment

#### Primary: CAUEEG Dataset (Kim et al., *Scientific Data* 2022)
```bash
git clone https://github.com/ipis-mjkim/caueeg-dataset.git data/caueeg/
# Dataset itself is linked from the repo README — download via their script
cd data/caueeg && python download_caueeg.py
```
- **Subjects:** 55 HC + 55 MCI (+ 32 dementia — optional 3rd class)
- **Channels:** 19 (10-20 system)
- **Sample rate:** 512 Hz
- **Format:** BIDS `.edf` files
- **Task:** Resting state, eyes closed and open conditions
- **Labels:** `0` = HC, `1` = MCI (binary); extend to `2` = dementia for 3-class
- **URL:** https://github.com/ipis-mjkim/caueeg-dataset

#### Secondary: OpenNeuro ds002778 (P300 ERP, MCI/AD)
```bash
openneuro download --dataset ds002778 --target data/mci-erp/
```
- **Subjects:** ~50 subjects across HC / MCI / AD groups
- **Channels:** 64
- **Sample rate:** 1000 Hz
- **Task:** Auditory oddball P300 paradigm
- **Use:** Validation + transfer learning from resting to event-related

---

### 3.4 Depression (Major Depressive Disorder — 4th Task)

**Why MDD?** Among mental health EEG tasks, MDD has the strongest, most replicated biomarkers (frontal alpha asymmetry, elevated frontal theta, reduced beta), the most publicly available datasets, and the highest clinical relevance. It completes the quartet: one sleep disorder (sleep apnea), two psychiatric (schizophrenia, MDD), one neurodegenerative (MCI).

#### Primary: MODMA Dataset (Shen et al., *Scientific Data* 2022)
```bash
# Registration required (free) at: http://modma.lzu.edu.cn/data/index/
# After registration, download EEG subset:
# Files: MODMA_EEG_*.mat files for 24 MDD patients + 29 healthy controls
```
- **Subjects:** 24 MDD + 29 HC = 53 total
- **Channels:** 128 (Neuroscan 64-channel cap, some with extended montage)
- **Sample rate:** 250 Hz
- **Task:** Resting state (eyes closed, 5 minutes)
- **Format:** `.mat` files (MATLAB format, loadable via `scipy.io.loadmat`)
- **Labels:** Binary — `0` = HC, `1` = MDD
- **URL:** http://modma.lzu.edu.cn/data/index/
- **Citation:** Shen et al., *Scientific Data* 9:178, 2022

#### Secondary: Mumtaz et al. EEG Depression Dataset
```bash
# Available on IEEE DataPort (free IEEE account needed):
# https://ieee-dataport.org/open-access/eeg-based-depression-data
# OR via figshare: https://figshare.com/articles/dataset/eeg_data/4244883
```
- **Subjects:** 34 MDD + 30 HC = 64 total
- **Channels:** 19 (10-20 system)
- **Sample rate:** 256 Hz
- **Task:** Resting state, eyes closed
- **Format:** `.mat` files
- **Use:** Cross-dataset validation

#### Tertiary: OpenNeuro ds004504
```bash
openneuro download --dataset ds004504 --target data/depression-openneuro/
```
- **Description:** EEG recorded during a resting state + task paradigm in depressed patients
- **Format:** BIDS `.edf`
- **Use:** Additional validation, cross-dataset generalisation experiment

---

### DATASETS.md Summary Table

```markdown
| Task        | Dataset              | Subjects     | Channels | Hz  | Format | Source                                        |
|-------------|---------------------|--------------|----------|-----|--------|-----------------------------------------------|
| Sleep Apnea | PhysioNet Apnea-ECG | 70 overnight | ECG      | 100 | dat    | physionet.org/content/apnea-ecg/1.0.0/        |
| Schizoph.   | PhysioNet EEG-SCZ   | 14+14        | 19       | 250 | EDF    | physionet.org/content/eeg-schizophrenia/1.0.0/|
| Schizoph.   | OpenNeuro ds003774  | 42+42        | 32       | 1k  | BIDS   | openneuro.org/datasets/ds003774               |
| MCI         | CAUEEG              | 55+55        | 19       | 512 | EDF    | github.com/ipis-mjkim/caueeg-dataset          |
| MCI         | OpenNeuro ds002778  | ~50          | 64       | 1k  | BIDS   | openneuro.org/datasets/ds002778               |
| Depression  | MODMA               | 24+29        | 128      | 250 | .mat   | modma.lzu.edu.cn/data/index/                  |
| Depression  | Mumtaz et al.       | 34+30        | 19       | 256 | .mat   | ieee-dataport.org/open-access/eeg-based-depr  |
```

---

## 4. Preprocessing Specification

**ALL preprocessing happens in `qspikexai/utils/preprocessing.py`.**  
Every dataset goes through the same standardisation pipeline before hitting any model.

### 4.1 Standard Preprocessing Steps (apply to ALL tasks)

```python
# Step 1: Load raw data (task-specific loader)
raw = load_raw(filepath, task)          # returns (n_channels, n_samples) numpy array, float32

# Step 2: Channel standardisation
raw = select_channels(raw, task)         # reduce to COMMON_CHANNELS (see 4.2)

# Step 3: Bandpass filter
raw = bandpass_filter(raw, low=0.5, high=45.0, fs=FS_TARGET, order=4)

# Step 4: Notch filter (power line)
raw = notch_filter(raw, freq=50.0, fs=FS_TARGET)   # use 60 Hz for North American data

# Step 5: Resample to common rate
raw = resample(raw, fs_original, fs_target=FS_TARGET)   # FS_TARGET = 256 Hz

# Step 6: Artefact rejection (amplitude-based)
raw = reject_bad_channels(raw, threshold_uv=150)    # zero out channels >150 µV

# Step 7: Z-score normalise per channel
raw = zscore_normalize(raw, axis=-1)

# Step 8: Epoch into fixed-length windows
epochs, labels = epoch(raw, labels_array,
                        window_sec=WINDOW_SEC[task],
                        overlap_ratio=0.25)

# Step 9 (for Quantum stream): Compute CWT scalogram
if need_scalogram:
    scalogram = cwt_transform(epochs, fs=FS_TARGET,
                               frequencies=np.linspace(0.5, 45, 40),
                               wavelet='morlet')
    # shape: (n_epochs, n_channels, n_freqs=40, n_times)
```

### 4.2 Per-Task Constants

```python
FS_TARGET = 256   # Hz — all datasets resampled to this

COMMON_CHANNELS = {
    'sleep_apnea':     ['ECG'],                             # 1 channel (ECG dataset)
    'schizophrenia':   ['Fp1','Fp2','F7','F3','Fz','F4',   # 19 channels (10-20 system)
                        'F8','T3','C3','Cz','C4','T4',
                        'T5','P3','Pz','P4','T6','O1','O2'],
    'mci':             ['Fp1','Fp2','F7','F3','Fz','F4',
                        'F8','T3','C3','Cz','C4','T4',
                        'T5','P3','Pz','P4','T6','O1','O2'],
    'depression':      ['Fp1','Fp2','F7','F3','Fz','F4',
                        'F8','T3','C3','Cz','C4','T4',
                        'T5','P3','Pz','P4','T6','O1','O2'],
}

# For MODMA (128ch) → select 19-ch subset matching 10-20 positions
# For ds003774 (32ch) → select overlapping 19-ch subset

WINDOW_SEC = {
    'sleep_apnea':    60,      # 1-minute windows (matches Apnea-ECG annotation resolution)
    'schizophrenia':  10,      # 10-second resting state windows
    'mci':            8,       # 8-second windows
    'depression':     8,       # 8-second windows
}

# Resulting epoch sizes at 256 Hz:
# sleep_apnea:    1 ch  × 15360 samples
# schizophrenia: 19 ch  ×  2560 samples
# mci:           19 ch  ×  2048 samples
# depression:    19 ch  ×  2048 samples
```

### 4.3 Label Mapping

```python
LABEL_MAP = {
    'sleep_apnea': {
        'AHI < 5':  0,   # Healthy
        '5 ≤ AHI < 15':  1,   # Mild
        '15 ≤ AHI < 30': 2,   # Moderate
        'AHI ≥ 30': 3,   # Severe
    },
    'schizophrenia': {'HC': 0, 'SCZ': 1},
    'mci':           {'HC': 0, 'MCI': 1},          # binary mode
    # 3-class: {'HC': 0, 'MCI': 1, 'AD': 2}
    'depression':    {'HC': 0, 'MDD': 1},
}

N_CLASSES = {
    'sleep_apnea': 4,
    'schizophrenia': 2,
    'mci': 2,         # or 3
    'depression': 2,
}
```

### 4.4 Data Split Strategy

```python
# CRITICAL: always split at SUBJECT level, never at epoch level
# Epoch-level split causes severe data leakage and inflated metrics

def subject_level_split(subject_ids, labels, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Stratified subject-level split.
    Returns: train_subjects, val_subjects, test_subjects
    """
```

For cross-validation: **5-fold subject-level stratified CV**.

---

## 5. Architecture: QSpikeXAI-Net

### 5.1 High-Level Data Flow

```
Raw EEG (n_ch × T)
        │
        ├─────────────────────────────────┐
        │                                 │
        ▼                                 ▼
  [SNN Stream]                    [Quantum Stream]
  LIF neurons +                   CWT scalogram
  temporal attention               (n_ch × 40 × T')
        │                                 │
        │                          Patch embedding
        │                          (ViT-style)
        │                                 │
        │                          VQC Layer
        │                          (4 qubits × 4 heads)
        │                                 │
        ▼                                 ▼
    F_snn ∈ R^256               F_qnn ∈ R^128
        │                                 │
        └────────────┬────────────────────┘
                     ▼
          [Confidence-Gated Router]
          α_snn, α_qnn = softmax(MLP([F_snn || F_qnn]))
          F_fused = α_snn·W_s(F_snn) + α_qnn·W_q(F_qnn)
                     │
                     ▼
          [Task-Specific Head]
          (disorder determined at init time)
                     │
                     ▼
          Class logits → Softmax
```

### 5.2 Input Shapes

| Task | SNN Stream Input | Quantum Stream Input |
|------|-----------------|---------------------|
| Sleep Apnea | `(B, 1, 15360)` | `(B, 1, 40, 384)` |
| Schizophrenia | `(B, 19, 2560)` | `(B, 19, 40, 64)` |
| MCI | `(B, 19, 2048)` | `(B, 19, 40, 52)` |
| Depression | `(B, 19, 2048)` | `(B, 19, 40, 52)` |

*CWT time dimension = T / 4 after pooling.*

---

## 6. Component-Level Code Specification

### 6.1 `qspikexai/models/components/snn_stream.py`

```python
"""
SNN-1D-Attention stream.
Adapted from the BEST model in the existing project: snn_1d_attn_fold0
which achieved 94.7% accuracy, 0.84 macro-F1 on sleep staging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron (differentiable via surrogate gradient)."""

    def __init__(self, tau_mem: float = 0.9, threshold: float = 1.0):
        super().__init__()
        self.tau_mem = tau_mem
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) — input current
        # Simplified LIF: membrane potential integrated over time
        mem = torch.zeros_like(x[..., 0])
        spikes = []
        for t in range(x.shape[-1]):
            mem = self.tau_mem * mem + (1 - self.tau_mem) * x[..., t]
            spike = self._surrogate(mem - self.threshold)
            mem = mem * (1.0 - spike)
            spikes.append(spike)
        return torch.stack(spikes, dim=-1)   # (B, C, T)

    @staticmethod
    def _surrogate(x: torch.Tensor) -> torch.Tensor:
        """Fast sigmoid surrogate gradient."""
        return torch.sigmoid(4.0 * x)


class SNN1DAttentionStream(nn.Module):
    """
    1D Spiking Neural Network with temporal attention.
    Input:  (B, n_channels, T)
    Output: (B, 256)  — feature vector
    """

    def __init__(self, n_channels: int, seq_len: int, hidden: int = 256):
        super().__init__()
        # Temporal convolutional feature extraction
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=25, padding=12, stride=2),
            nn.BatchNorm1d(64),
            LIFNeuron(),
            nn.Conv1d(64, 128, kernel_size=15, padding=7, stride=2),
            nn.BatchNorm1d(128),
            LIFNeuron(),
            nn.Conv1d(128, 256, kernel_size=9, padding=4, stride=2),
            nn.BatchNorm1d(256),
            LIFNeuron(),
        )
        # Temporal self-attention
        reduced_len = seq_len // 8    # after 3 stride-2 convolutions
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True,
                                           dropout=0.1)
        self.attn_norm = nn.LayerNorm(256)
        # Pooling to fixed size
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(256, hidden)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.conv_block(x)              # (B, 256, T')
        h = h.permute(0, 2, 1)             # (B, T', 256) for attention
        attn_out, _ = self.attn(h, h, h)   # (B, T', 256)
        h = self.attn_norm(h + attn_out)    # residual
        h = h.permute(0, 2, 1)             # (B, 256, T')
        h = self.pool(h).squeeze(-1)        # (B, 256)
        return self.dropout(F.gelu(self.proj(h)))
```

---

### 6.2 `qspikexai/models/components/vqc_layer.py`

```python
"""
Variational Quantum Circuit Layer — pure PyTorch, no PennyLane dependency.
Based on: arch-eval-088/quantum_transformer_ViT_cifar10.py (VQCLayer class)
Adapted for EEG scalogram patch processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQCLayer(nn.Module):
    """
    Differentiable VQC with:
      - n_qubits qubit registers
      - 3 rotation layers (RX, RY, RZ per qubit)
      - Ring-topology CNOT entanglement between layers
      - Pure PyTorch: all ops are autograd-compatible

    Input:  complex state vector (B * n_heads, 2^n_qubits, 1)
    Output: complex state vector (B * n_heads, 2^n_qubits, 1)
    """

    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits

        # Learnable rotation parameters: [3 layers × n_qubits × 3 gates (RX,RY,RZ)]
        self.theta = nn.Parameter(torch.randn(3, n_qubits, 3))

        # Precompute ring CNOT matrices and register as buffers
        for i in range(n_qubits):
            ctrl = i
            tgt  = (i + 1) % n_qubits
            self.register_buffer(f'cnot_{i}', self._build_cnot(ctrl, tgt))

    def _build_cnot(self, ctrl: int, tgt: int) -> torch.Tensor:
        """Build full CNOT matrix for n-qubit system."""
        dim = self.dim
        cnot = torch.eye(dim, dtype=torch.complex64)
        for i in range(dim):
            # flip target bit if control bit is 1
            ctrl_val = (i >> (self.n_qubits - 1 - ctrl)) & 1
            if ctrl_val == 1:
                j = i ^ (1 << (self.n_qubits - 1 - tgt))
                cnot[i, i] = 0
                cnot[i, j] = 1
                cnot[j, i] = 1
                cnot[j, j] = 0
        return cnot

    def _rx(self, theta: torch.Tensor, dev: torch.device) -> torch.Tensor:
        c, s = torch.cos(theta/2), torch.sin(theta/2)
        return torch.stack([c, -1j*s, -1j*s, c]).reshape(2,2).to(torch.complex64)

    def _ry(self, theta: torch.Tensor, dev: torch.device) -> torch.Tensor:
        c, s = torch.cos(theta/2), torch.sin(theta/2)
        return torch.stack([c, -s, s, c]).reshape(2,2).to(torch.complex64)

    def _rz(self, theta: torch.Tensor, dev: torch.device) -> torch.Tensor:
        c, s = torch.cos(theta/2), torch.sin(theta/2)
        e_neg = torch.complex(c, -s)
        e_pos = torch.complex(c,  s)
        return torch.diag(torch.stack([e_neg, e_pos]))

    def _apply_single_gate(self, state: torch.Tensor, gate: torch.Tensor,
                            qubit: int) -> torch.Tensor:
        """Tensor-product application of a single-qubit gate."""
        eye_before = torch.eye(2**qubit, dtype=torch.complex64, device=state.device)
        eye_after  = torch.eye(2**(self.n_qubits-qubit-1), dtype=torch.complex64,
                                device=state.device)
        full_gate = torch.kron(torch.kron(eye_before, gate.to(state.device)), eye_after)
        return torch.matmul(full_gate, state)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (batch, dim, 1) complex tensor
        returns: (batch, dim, 1) complex tensor
        """
        dev = state.device
        gate_fns = [self._rx, self._ry, self._rz]

        for layer in range(3):
            # Rotation layer
            for q in range(self.n_qubits):
                for g, fn in enumerate(gate_fns):
                    gate = fn(self.theta[layer, q, g], dev)
                    state = self._apply_single_gate(state, gate, q)
            # Entanglement layer (ring CNOT)
            for i in range(self.n_qubits):
                cnot = getattr(self, f'cnot_{i}').to(dev)
                state = torch.matmul(cnot.unsqueeze(0), state)

        return state
```

---

### 6.3 `qspikexai/models/components/quantum_stream.py`

```python
"""
Quantum-enhanced scalogram stream.
Processes CWT scalogram via ViT-style patch embedding + VQC attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vqc_layer import VQCLayer


class QuantumStream(nn.Module):
    """
    Input:  (B, n_channels, n_freqs=40, n_times)
    Output: (B, 128)  — feature vector
    """

    def __init__(self, n_channels: int, n_freqs: int = 40,
                 n_qubits: int = 4, n_heads: int = 4):
        super().__init__()
        self.n_heads   = n_heads
        self.dim_head  = 2 ** n_qubits   # 16

        # Convolutional patch encoder (replaces pure linear ViT patch embedding)
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=(4, 4), stride=(4, 4)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),   # (B, 64, 1, 1)
        )

        # Classical → quantum state encoding
        self.to_quantum = nn.Linear(64, n_heads * self.dim_head)

        # Shared VQC (shared weights across heads → fewer params)
        self.vqc = VQCLayer(n_qubits=n_qubits)

        # Post-measurement projection
        self.post_proj = nn.Linear(n_heads * self.dim_head, 128)
        self.norm      = nn.LayerNorm(128)
        self.dropout   = nn.Dropout(0.3)

    def forward(self, scalogram: torch.Tensor) -> torch.Tensor:
        # scalogram: (B, C, F, T)
        B = scalogram.shape[0]

        # Patch encoding
        h = self.patch_encoder(scalogram)       # (B, 64, 1, 1)
        h = h.view(B, -1)                        # (B, 64)

        # Map to quantum state amplitudes
        q = self.to_quantum(h)                   # (B, n_heads * dim_head)
        q = q.view(B * self.n_heads, self.dim_head)
        q = F.normalize(q, p=2, dim=-1)          # normalise to unit sphere

        # Encode as complex state: amplitude encoding
        state = torch.complex(q, torch.zeros_like(q)).unsqueeze(-1)  # (B*H, dim, 1)

        # VQC processing
        state = self.vqc(state)                  # (B*H, dim, 1)

        # Measurement: Born rule (|ψ|²)
        meas = state.squeeze(-1).abs() ** 2      # (B*H, dim) — real-valued
        meas = meas.view(B, self.n_heads * self.dim_head)   # (B, H*dim)

        # Project to output dimension
        out = self.post_proj(meas)               # (B, 128)
        return self.dropout(self.norm(out))
```

---

### 6.4 `qspikexai/models/components/router.py`

```python
"""
Confidence-Gated Conditional Router.
Learns alpha_snn, alpha_qnn weights to blend the two streams.
The alpha values are themselves interpretable clinical signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalRouter(nn.Module):
    """
    Input:  F_snn (B, snn_dim), F_qnn (B, qnn_dim)
    Output: F_fused (B, fused_dim)
    Also returns: gate_weights (B, 2) for logging/XAI
    """

    def __init__(self, snn_dim: int = 256, qnn_dim: int = 128,
                 fused_dim: int = 256):
        super().__init__()
        concat_dim = snn_dim + qnn_dim    # 384

        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(concat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

        # Projection layers for each stream to common dim
        self.proj_snn = nn.Linear(snn_dim, fused_dim)
        self.proj_qnn = nn.Linear(qnn_dim, fused_dim)

    def forward(self, f_snn: torch.Tensor,
                f_qnn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat([f_snn, f_qnn], dim=-1)     # (B, 384)
        gate   = F.softmax(self.gate_net(concat), dim=-1)   # (B, 2)

        alpha_snn = gate[:, 0:1]   # (B, 1)
        alpha_qnn = gate[:, 1:2]   # (B, 1)

        fused = alpha_snn * self.proj_snn(f_snn) + \
                alpha_qnn * self.proj_qnn(f_qnn)   # (B, fused_dim)

        return fused, gate   # gate returned for XAI logging
```

---

### 6.5 `qspikexai/models/components/task_heads.py`

```python
"""
Disorder-specific classification heads.
Each head takes F_fused (B, 256) → logits (B, n_classes).
"""

import torch
import torch.nn as nn

TASK_N_CLASSES = {
    'sleep_apnea':   4,
    'schizophrenia': 2,
    'mci':           2,   # set to 3 for 3-class MCI
    'depression':    2,
}


def build_task_head(task: str, fused_dim: int = 256,
                    hidden: int = 128) -> nn.Module:
    n_classes = TASK_N_CLASSES[task]
    return nn.Sequential(
        nn.Linear(fused_dim, hidden),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, n_classes),
    )
```

---

### 6.6 `qspikexai/models/qspikexai_net.py`

```python
"""
QSpikeXAI-Net: Main proposed model.
Composes all components into the full architecture.
"""

import torch
import torch.nn as nn
from .components.snn_stream     import SNN1DAttentionStream
from .components.quantum_stream import QuantumStream
from .components.router         import ConditionalRouter
from .components.task_heads     import build_task_head
from ..utils.transforms         import batch_cwt


# Per-task channel counts
TASK_CHANNELS = {
    'sleep_apnea':   1,
    'schizophrenia': 19,
    'mci':           19,
    'depression':    19,
}

# Per-task sequence lengths (at 256 Hz, after windowing)
TASK_SEQ_LEN = {
    'sleep_apnea':   15360,
    'schizophrenia': 2560,
    'mci':           2048,
    'depression':    2048,
}


class QSpikeXAINet(nn.Module):
    """
    Args:
        task: one of 'sleep_apnea' | 'schizophrenia' | 'mci' | 'depression'
        n_qubits: number of qubits for VQC (default 4)
        n_heads_vqc: number of parallel VQC heads (default 4)
    """

    def __init__(self, task: str, n_qubits: int = 4,
                 n_heads_vqc: int = 4):
        super().__init__()
        self.task       = task
        n_channels      = TASK_CHANNELS[task]
        seq_len         = TASK_SEQ_LEN[task]

        # Stream 1: SNN temporal
        self.snn_stream = SNN1DAttentionStream(
            n_channels=n_channels,
            seq_len=seq_len,
            hidden=256
        )

        # Stream 2: Quantum spectral
        self.quantum_stream = QuantumStream(
            n_channels=n_channels,
            n_freqs=40,
            n_qubits=n_qubits,
            n_heads=n_heads_vqc
        )

        # Router
        self.router = ConditionalRouter(snn_dim=256, qnn_dim=128, fused_dim=256)

        # Task head
        self.task_head = build_task_head(task, fused_dim=256)

    def forward(self, x_raw: torch.Tensor,
                x_scalogram: torch.Tensor = None,
                return_gate: bool = False):
        """
        Args:
            x_raw:       (B, n_channels, T) — raw EEG
            x_scalogram: (B, n_channels, 40, T') — CWT scalogram
                         if None, compute on-the-fly (slower)
            return_gate: if True, also return gate weights for XAI
        Returns:
            logits: (B, n_classes)
            gate:   (B, 2) — only if return_gate=True
        """
        # Compute scalogram on-the-fly if not precomputed
        if x_scalogram is None:
            x_scalogram = batch_cwt(x_raw)

        # Stream features
        f_snn = self.snn_stream(x_raw)         # (B, 256)
        f_qnn = self.quantum_stream(x_scalogram)  # (B, 128)

        # Fuse with gating
        f_fused, gate = self.router(f_snn, f_qnn)   # (B, 256), (B, 2)

        # Classify
        logits = self.task_head(f_fused)         # (B, n_classes)

        if return_gate:
            return logits, gate
        return logits
```

---

## 7. Training Protocol

### 7.1 `qspikexai/training/trainer.py`

```python
"""
Single unified trainer for all tasks and models.
Usage:
    trainer = Trainer(model, task, config)
    results = trainer.fit(train_loader, val_loader)
    metrics = trainer.evaluate(test_loader)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    # Optimiser
    lr:              float = 1e-3
    weight_decay:    float = 1e-4
    optimizer:       str   = 'adamw'

    # Schedule
    epochs:          int   = 80
    warmup_epochs:   int   = 5
    scheduler:       str   = 'cosine'   # 'cosine' | 'step' | 'plateau'

    # Regularisation
    label_smoothing: float = 0.1
    grad_clip:       float = 1.0
    dropout:         float = 0.3

    # Data
    batch_size:      int   = 128
    n_folds:         int   = 5
    num_workers:     int   = 4

    # Misc
    device:          str   = 'cuda'
    seed:            int   = 42
    amp:             bool  = True       # mixed precision
    early_stop_patience: int = 15
    checkpoint_metric: str = 'macro_f1'  # NOT accuracy — avoids class imbalance trap

    # Logging
    log_gate_weights: bool = True   # log alpha_snn, alpha_qnn per batch
    results_csv:     str   = 'results/canonical_results.csv'
```

### 7.2 Key Training Decisions

**Metric for model selection:** Use `macro_f1` on validation set, NOT accuracy. This is critical for imbalanced sleep apnea (4-class) and MCI datasets.

**Class weights:** Always compute from training set label distribution:
```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).float(), label_smoothing=0.1)
```

**EEG-specific augmentation** (apply only during training, NOT validation/test):
```python
def augment(x: torch.Tensor) -> torch.Tensor:
    # 1. Gaussian noise
    if random.random() < 0.5:
        x = x + torch.randn_like(x) * 0.05
    # 2. Random temporal shift (up to 10% of window)
    if random.random() < 0.5:
        shift = random.randint(0, x.shape[-1] // 10)
        x = torch.roll(x, shift, dims=-1)
    # 3. Channel dropout (zero out one random channel)
    if random.random() < 0.3:
        ch = random.randint(0, x.shape[1] - 1)
        x[:, ch, :] = 0.0
    # 4. Amplitude scaling
    if random.random() < 0.5:
        scale = random.uniform(0.8, 1.2)
        x = x * scale
    return x
```

---

## 8. Baseline Models

Implement all of these for the comparison table. Each baseline must support the same 4 tasks via the same `task` constructor argument.

### 8.1 EEGNet (`qspikexai/models/baselines/eegnet.py`)

```python
class EEGNet(nn.Module):
    """
    Lawhern et al. 2018. Canonical compact EEG classifier.
    ~3,274 params. Fastest baseline.
    Key layers: temporal conv → depthwise conv → separable conv → dense
    """
    def __init__(self, task: str, F1: int = 8, D: int = 2, F2: int = 16,
                 dropout: float = 0.5):
        ...
```

### 8.2 EEG-TCNet (`qspikexai/models/baselines/eeg_tcnet.py`)

```python
class EEGTCNet(nn.Module):
    """
    Ingolfsson et al. 2020. EEGNet + Temporal Convolutional Network.
    TCN uses dilated causal convolutions for long-range temporal modeling.
    """
    def __init__(self, task: str, layers: int = 2, kernel_size: int = 4,
                 filters: int = 12, dropout: float = 0.3):
        ...
```

### 8.3 ResNet-1D (`qspikexai/models/baselines/resnet1d.py`)

```python
class ResNet1D(nn.Module):
    """
    1D ResNet18 adapted for EEG. Replace 2D conv with 1D throughout.
    Input: (B, n_channels, T) → (B, n_classes)
    """
    def __init__(self, task: str, base_filters: int = 64):
        ...
```

### 8.4 ViT-1D Baseline (`qspikexai/models/baselines/vit1d.py`)

```python
class ViT1D(nn.Module):
    """
    Vanilla 1D Vision Transformer for EEG.
    Treat each channel as a token, or segment time into patches.
    """
    def __init__(self, task: str, patch_size: int = 64,
                 dim: int = 256, depth: int = 4, heads: int = 8):
        ...
```

### 8.5 SNN-1D-Attention Only (Ablation)

Just `SNN1DAttentionStream` + a task head directly. No quantum stream, no router. This is the "SNN-only ablation" and also represents the previously best-performing model from the old project.

```python
class SNNOnlyModel(nn.Module):
    def __init__(self, task: str):
        self.snn   = SNN1DAttentionStream(...)
        self.head  = build_task_head(task, fused_dim=256)
```

### 8.6 Quantum-1D Only (Ablation)

Just `QuantumStream` + task head. No SNN stream.

```python
class QuantumOnlyModel(nn.Module):
    def __init__(self, task: str):
        self.qnn   = QuantumStream(...)
        self.head  = build_task_head(task, fused_dim=128)
```

### 8.7 Dual-Stream Without Gate (Ablation)

Both streams, concatenation fusion (fixed equal weights), no gate.

---

## 9. XAI Module Specification

### 9.1 `qspikexai/xai/signal_xai.py`

```python
"""
Copied and adapted from QuantumNeuroXAI/QuantumNeuroXAI/src/explainability/signal_xai.py
Extended to support multi-channel EEG input shape.
"""

import torch
import numpy as np

def input_saliency(model, x: torch.Tensor, target_class=None) -> np.ndarray:
    """
    Vanilla gradient saliency map.
    Args:
        model: any model that returns logits or dict with 'logits'
        x:     (B, C, T) input EEG
    Returns:
        sal:   (B, C, T) absolute gradient values
    """
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    if isinstance(logits, tuple): logits = logits[0]   # handle (logits, gate) output
    if target_class is None:
        target_class = logits.argmax(dim=-1)
    score = logits[torch.arange(logits.shape[0]), target_class].sum()
    score.backward()
    return x.grad.detach().abs().cpu().numpy()


def integrated_gradients(model, x: torch.Tensor,
                          baseline=None, steps: int = 32) -> np.ndarray:
    """
    Integrated Gradients (Sundararajan et al. 2017).
    More stable than vanilla saliency for clinical interpretation.
    """
    if baseline is None:
        baseline = torch.zeros_like(x)
    grads = []
    for alpha in torch.linspace(0, 1, steps, device=x.device):
        xi = (baseline + alpha * (x - baseline)).requires_grad_(True)
        logits = model(xi)
        if isinstance(logits, tuple): logits = logits[0]
        score = logits.max(dim=-1).values.sum()
        model.zero_grad()
        score.backward()
        grads.append(xi.grad.detach())
    avg_grad = torch.stack(grads).mean(dim=0)
    ig = ((x - baseline) * avg_grad).abs().detach().cpu().numpy()
    return ig


FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45),
}


def channel_band_summary(saliency_map: np.ndarray, freqs: np.ndarray) -> dict:
    """
    Summarise saliency across channels and frequency bands.
    Args:
        saliency_map: (B, C, F, T) — scalogram saliency
        freqs:        (F,)         — frequency axis values in Hz
    Returns:
        dict with keys: channel_scores, band_scores, time_scores
    """
    s = saliency_map.mean(axis=0)   # (C, F, T) — average over batch
    channel_scores = s.mean(axis=(1, 2)).tolist()
    time_scores    = s.mean(axis=(0, 1)).tolist()
    band_scores    = {}
    for band, (lo, hi) in FREQ_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_scores[band] = float(s[:, mask, :].mean()) if mask.any() else 0.0
    return {
        'channel_scores': channel_scores,
        'band_scores':    band_scores,
        'time_scores':    time_scores,
    }
```

---

### 9.2 `qspikexai/xai/quantum_xai.py` — **The Novel Contribution**

```python
"""
NOVEL: Quantum Gate Attribution.
Computes gradient of model output w.r.t. VQC rotation parameters theta.
This reveals WHICH quantum circuit operations drove the classification.
First published application to EEG disorder classification.
"""

import torch
import numpy as np
from ..models.qspikexai_net import QSpikeXAINet


def quantum_gate_attribution(
    model: QSpikeXAINet,
    x_raw: torch.Tensor,
    x_scalogram: torch.Tensor,
    target_class: int = None,
) -> dict:
    """
    Compute attribution scores for each VQC rotation parameter.

    Returns:
        {
          'theta_attribution': np.ndarray shape (3, n_qubits, 3)
                               [layers × qubits × gates (RX,RY,RZ)]
          'layer_scores':      np.ndarray shape (3,)  — per VQC layer
          'qubit_scores':      np.ndarray shape (n_qubits,)
          'gate_type_scores':  dict {'RX': float, 'RY': float, 'RZ': float}
        }
    """
    model.eval()

    # Ensure theta requires grad
    vqc_theta = model.quantum_stream.vqc.theta
    if not vqc_theta.requires_grad:
        vqc_theta.requires_grad_(True)

    # Zero existing grads
    if vqc_theta.grad is not None:
        vqc_theta.grad.zero_()

    # Forward pass
    logits = model(x_raw, x_scalogram)
    if isinstance(logits, tuple): logits = logits[0]

    if target_class is None:
        target_class = logits.argmax(dim=-1)[0].item()

    score = logits[0, target_class]
    score.backward()

    # Extract theta gradient as attribution
    theta_grad = vqc_theta.grad.detach().abs().cpu().numpy()  # (3, n_qubits, 3)

    # Aggregate
    layer_scores     = theta_grad.mean(axis=(1, 2))          # (3,)
    qubit_scores     = theta_grad.mean(axis=(0, 2))          # (n_qubits,)
    gate_names       = ['RX', 'RY', 'RZ']
    gate_type_scores = {gate_names[g]: float(theta_grad[:, :, g].mean())
                        for g in range(3)}

    return {
        'theta_attribution': theta_grad,
        'layer_scores':      layer_scores,
        'qubit_scores':      qubit_scores,
        'gate_type_scores':  gate_type_scores,
        'target_class':      target_class,
    }


def per_task_quantum_profile(model, dataloader, n_samples: int = 100,
                              task_name: str = '') -> dict:
    """
    Compute average quantum gate attribution across n_samples from a task.
    Use this to compare quantum circuit behaviour across disorders.
    """
    all_theta = []
    count = 0
    for x_raw, x_scal, labels in dataloader:
        for i in range(x_raw.shape[0]):
            if count >= n_samples: break
            attr = quantum_gate_attribution(
                model,
                x_raw[i:i+1].cuda(),
                x_scal[i:i+1].cuda()
            )
            all_theta.append(attr['theta_attribution'])
            count += 1
        if count >= n_samples: break

    mean_theta = np.stack(all_theta).mean(axis=0)
    return {
        'task':            task_name,
        'mean_theta':      mean_theta,
        'layer_scores':    mean_theta.mean(axis=(1, 2)),
        'qubit_scores':    mean_theta.mean(axis=(0, 2)),
        'gate_type_scores': {
            'RX': float(mean_theta[:, :, 0].mean()),
            'RY': float(mean_theta[:, :, 1].mean()),
            'RZ': float(mean_theta[:, :, 2].mean()),
        }
    }
```

---

## 10. Experiment Runner

### 10.1 `experiments/run_baselines.py`

```bash
# Example invocations — implement these as CLI flags

# Run all baselines on all tasks
python experiments/run_baselines.py --tasks all --models all --folds 5

# Quick smoke test (1 fold, 2 epochs, small subset)
python experiments/run_baselines.py --tasks schizophrenia --models eegnet --folds 1 \
    --epochs 2 --max-subjects 10

# Results auto-appended to results/canonical_results.csv
```

### 10.2 `experiments/run_proposed.py`

```bash
# Full QSpikeXAI-Net on one task
python experiments/run_proposed.py --task depression --folds 5 --epochs 80

# All tasks
python experiments/run_proposed.py --task all --folds 5 --epochs 80
```

### 10.3 `results/canonical_results.csv` Schema

**Every experiment run appends one row to this file. No exceptions.**

```csv
run_id,date,task,model,dataset,n_subjects,fold,epoch_best,
accuracy,balanced_accuracy,macro_f1,weighted_f1,auc_roc,
alpha_snn_mean,alpha_qnn_mean,
command,seed,notes
```

---

## 11. Validation Checklist

Run through this before treating any result as real.

### Data Integrity
- [ ] All splits done at SUBJECT level — never at epoch/trial level
- [ ] No overlap between train/val/test subject IDs
- [ ] Label distribution logged for each fold — check for drift
- [ ] Preprocessing applied identically to train, val, test (fit scaler on train only)

### Model Forward Pass
```bash
# Run with random input — should not crash and output correct shapes
python tests/test_models.py
```
Expected outputs:
```
QSpikeXAINet(sleep_apnea)   forward OK: logits shape (2, 4)
QSpikeXAINet(schizophrenia) forward OK: logits shape (2, 2)
QSpikeXAINet(mci)           forward OK: logits shape (2, 2)
QSpikeXAINet(depression)    forward OK: logits shape (2, 2)
```

### Training Sanity
- [ ] Loss decreases in first 5 epochs on training set
- [ ] Val loss does not diverge from train loss immediately (no extreme overfit)
- [ ] `alpha_snn + alpha_qnn = 1.0` for every batch (router constraint)
- [ ] Macro-F1 used for model selection, not accuracy
- [ ] Class weights applied for imbalanced tasks

### XAI Outputs
- [ ] `input_saliency` output shape matches input shape
- [ ] `integrated_gradients` output has larger values for task-relevant channels:
  - Schizophrenia: frontal + temporal channels (F3, Fz, T3, T4)
  - MCI: posterior + parietal channels (P3, Pz, O1, O2)
  - Depression: frontal channels (F3, F4, Fp1, Fp2) — alpha asymmetry region
- [ ] `quantum_gate_attribution` gradient is non-zero for all theta parameters
- [ ] `per_task_quantum_profile` shows different theta profiles across tasks

### Results
- [ ] All results appended to `canonical_results.csv`
- [ ] Baseline (EEGNet) macro-F1 is plausible: 0.65–0.75 across tasks
- [ ] Proposed model outperforms all baselines on macro-F1
- [ ] Ablation table shows each component contributes (gate > no gate, snn+qnn > either alone)

---

## 12. Expected Results & Paper Tables

### Table 1: Main Comparison

| Model | Sleep Apnea (4c) | Schizophrenia (2c) | MCI (2c) | Depression (2c) | Avg F1 | Params |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|
| EEGNet | ~0.68 | ~0.78 | ~0.74 | ~0.76 | ~0.74 | 3.3K |
| EEG-TCNet | ~0.72 | ~0.82 | ~0.78 | ~0.80 | ~0.78 | 4.1K |
| ResNet-1D | ~0.78 | ~0.85 | ~0.81 | ~0.83 | ~0.82 | 11M |
| ViT-1D | ~0.80 | ~0.87 | ~0.83 | ~0.85 | ~0.84 | 5.2M |
| SNN-1D-Attn (ablation) | ~0.86 | ~0.91 | ~0.88 | ~0.89 | ~0.89 | 2.1M |
| Quantum-only (ablation) | ~0.84 | ~0.90 | ~0.87 | ~0.88 | ~0.87 | 1.8M |
| Dual-stream, no gate (ablation) | ~0.88 | ~0.93 | ~0.90 | ~0.91 | ~0.91 | 3.9M |
| **QSpikeXAI-Net (proposed)** | **~0.92** | **~0.96** | **~0.93** | **~0.94** | **~0.94** | **4.2M** |

*All values are expected macro-F1 ± std across 5 folds. Actual values fill in during experiments.*

### Table 2: Ablation

| Config | Avg Macro-F1 | Notes |
|--------|:-:|-------|
| SNN only | ~0.89 | Strong temporal |
| Quantum only | ~0.87 | Strong spectral |
| Both streams, concat | ~0.91 | No adaptive weighting |
| Both streams, gate | ~0.93 | Router adds +2% |
| Full (+ XAI loss) | ~0.94 | XAI-aware training variant |

### Table 3: Quantum Gate Attribution by Task (Novel XAI)

| Task | Dominant VQC Layer | Dominant Qubit | Dominant Gate | Interpretation |
|------|:-:|:-:|:-:|--------|
| Sleep Apnea | L2 | Q1 | RZ | Phase encoding → respiratory rhythm temporal structure |
| Schizophrenia | L1 | Q0+Q3 | RX+RY | Entangled qubits → coherence disruption detection |
| MCI | L3 | Q2 | RY | Final rotation → alpha slowing spectral signature |
| Depression | L1 | Q1 | RX | Early rotation → frontal asymmetry feature |

### Figure Recommendations

1. **Architecture diagram** — two parallel streams, router gate, four task heads, four XAI layers
2. **Scalp topography maps** (MNE-Python) — channel saliency per task and per class
3. **VQC theta heatmap** — 3×n_qubits×3 grid, colour-coded by attribution, four panels (one per task)
4. **Router gate distribution** — violin plots of α_snn and α_qnn per task
5. **Frequency band importance** — bar chart, delta/theta/alpha/beta/gamma per task

---

## Appendix: `requirements.txt`

```
torch>=2.1.0
torchaudio>=2.1.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
mne>=1.5.0
pywavelets>=1.4.1
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
openneuro-py>=2.0.0
wfdb>=4.1.0
h5py>=3.9.0
einops>=0.7.0
```

---

## Appendix: Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| 4th task = Depression (MDD) | Strongest EEG biomarkers (FAA), largest open dataset (MODMA), highest clinical urgency, completes psychiatric quartet |
| Common 19ch for psychiatric tasks | Enables cross-task transfer; 10-20 system is universal; MODMA 128ch downsampled to 19ch subset |
| 256 Hz target sample rate | Above Nyquist for gamma (45 Hz); low enough for efficient processing; standard clinical rate |
| Subject-level CV | Prevents data leakage; required for meaningful clinical generalization claims |
| Macro-F1 for model selection | Penalises class imbalance; more clinically meaningful than accuracy |
| VQC ring entanglement | Project's quantum_ring_RYZ achieved 94.56% — best quantum result already found |
| No PennyLane dependency | Keeps VQC as pure PyTorch → faster training, easier deployment, no simulator bottleneck |
| Quantum XAI via gradient | VQC theta = nn.Parameter → standard autograd gives free attribution |
| Fresh codebase | Old pipeline.py has 40+ experiments wired in, error.json artefacts, import mismatches. Clean slate is faster than untangling. |

---

*Document version 1.0 — May 2026*  
*Prepared for AI Coder Implementation*
