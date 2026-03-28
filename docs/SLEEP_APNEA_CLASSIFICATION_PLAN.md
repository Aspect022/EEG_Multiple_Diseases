# 🫁 Sleep Apnea Severity Classification Pipeline - Strategic Implementation Plan

**Document Version:** 1.0  
**Created:** March 20, 2026  
**Status:** Planning Phase  

---

## 📋 Executive Summary

This document outlines a comprehensive plan to implement a **4-class sleep apnea severity classification pipeline** using the **SHHS (Sleep Heart Health Study) dataset**, running parallel to the existing BOAS sleep staging experiments.

### Key Objectives

1. **4-Class Classification:** Healthy / Mild / Moderate / Severe sleep apnea
2. **Multiple Architectures:** Custom CNN, ResNet18 Transfer Learning, Hybrid ViT+BiLSTM
3. **Self-Supervised Pretraining:** All models with SSL pretraining
4. **Cross-Dataset Validation:** Secondary dataset for generalization testing
5. **Clean Separation:** Isolated from existing BOAS sleep staging pipeline

---

## 🧠 Problem Understanding

### Current State Analysis

The existing codebase (`D:\Projects\AI-Projects\EEG\`) contains:
- **BOAS Sleep Staging Pipeline:** 5-class sleep stage classification (W/N1/N2/N3/REM)
- **Multiple Model Architectures:** SNN, Quantum CNN, Swin Transformer, ViT, Fusion models
- **Established Patterns:** Dataset loaders, training loops, evaluation metrics
- **Infrastructure:** Precomputed scalogram caching, mixed precision training, gradient accumulation

### Key Differences: Sleep Staging vs. Apnea Classification

| Aspect | Sleep Staging (BOAS) | Apnea Classification (SHHS) |
|--------|---------------------|----------------------------|
| **Task** | 5-class sleep stage per 30s epoch | 4-class severity per recording/night |
| **Input** | 30-second EEG epochs | Full-night PSG or aggregated epochs |
| **Labels** | W, N1, N2, N3, REM | Healthy, Mild, Moderate, Severe |
| **Dataset** | BOAS (ds005555) | SHHS + secondary dataset |
| **Sequence** | Single epoch classification | Sequence modeling or aggregation |

### Critical Design Decisions

1. **Classification Granularity:** Per-recording (AHI-based) vs. Per-epoch (apnea event detection)
2. **Input Representation:** Raw EEG vs. Scalograms vs. Multi-modal PSG
3. **Sequence Modeling:** How to handle temporal dependencies across a full night
4. **Self-Supervised Pretraining:** Which SSL method (MAE, SimCLR, BYOL) for EEG

---

## ❓ Critical Questions (Resolved for Planning)

Based on standard sleep apnea classification research practices:

| Question | Decision | Rationale |
|----------|----------|-----------|
| **Classification unit?** | **Per-recording (night-level)** | AHI is computed per night; matches clinical practice |
| **Input length?** | **Full night or 30-min windows** | Captures sleep architecture; manageable sequence length |
| **Apnea severity definition?** | **AHI-based thresholds** | Standard clinical definition (see Section 4) |
| **Secondary dataset?** | **PhysioNet Apnea-ECG + MESA Sleep** | Publicly available, complementary demographics |
| **SSL method?** | **Masked Autoencoding (MAE)** | Proven effective for EEG; matches ViT architecture |

---

## ⚡ Strategy Options

### Option 1: **Modular Micro-Pipeline** (Recommended)

**Core Idea:** Create a standalone `sleep_apnea/` directory with self-contained modules that mirror the existing structure but operate independently.

**Pros:**
- ✅ Clean separation from BOAS experiments
- ✅ Easy to run independently or in parallel
- ✅ Minimal risk of breaking existing code
- ✅ Clear ownership and maintenance boundaries

**Cons:**
- ⚠️ Some code duplication (dataset patterns, training loops)
- ⚠️ Separate dependency management if needed

**When to Use:** When isolation is critical and experiments should not interfere.

---

### Option 2: **Unified Pipeline with Task Switching**

**Core Idea:** Extend existing `pipeline.py` with `--task` flag (`sleep_staging` vs `apnea_classification`).

**Pros:**
- ✅ Shared infrastructure (trainers, utilities, models)
- ✅ Single command interface
- ✅ Easier comparison between tasks

**Cons:**
- ⚠️ Increased complexity in existing pipeline
- ⚠️ Risk of breaking BOAS experiments
- ⚠️ Configuration conflicts

**When to Use:** When frequent comparison between tasks is needed.

---

### Option 3: **Hybrid: Shared Core, Task-Specific Wrappers**

**Core Idea:** Extract common components into `src/core/`, create task-specific runners (`run_boas.py`, `run_apnea.py`).

**Pros:**
- ✅ Best of both worlds: shared infrastructure + isolation
- ✅ Clean architecture for future tasks
- ✅ Refactoring opportunity

**Cons:**
- ⚠️ Requires refactoring existing code
- ⚠️ More upfront work

**When to Use:** Long-term maintainability is a priority.

---

## 🏆 Selected Strategy: **Option 1 (Modular Micro-Pipeline)**

**Justification:**

1. **Risk Mitigation:** BOAS experiments are already running; isolation prevents disruption
2. **Parallel Development:** Can iterate on apnea pipeline without affecting sleep staging
3. **Clear Boundaries:** Easier to share/publish apnea-specific code independently
4. **Proven Pattern:** Matches successful structure of existing `snn_1d/`, `quantum/`, `snn/` modules

**What We're Optimizing For:**
- ✅ Experiment isolation and reproducibility
- ✅ Fast iteration on apnea-specific features
- ✅ Clean separation for potential publication

**What We're Sacrificing:**
- ⚠️ Some code duplication (acceptable for clarity)
- ⚠️ Slightly more complex overall structure

---

## 🪜 Execution Plan

### Phase 1: Project Structure Setup (Day 1)

**Goal:** Create isolated directory structure for sleep apnea pipeline.

```
D:\Projects\AI-Projects\EEG\
├── sleep_apnea/                    # NEW: Apnea classification pipeline
│   ├── __init__.py
│   ├── README.md
│   ├── run_apnea.py                # Main entry point
│   ├── configs/
│   │   ├── default_config.yaml
│   │   ├── cnn_baseline.yaml
│   │   ├── resnet18_transfer.yaml
│   │   └── vit_bilstm_hybrid.yaml
│   ├── data/
│   │   ├── __init__.py
│   │   ├── shhs_dataset.py         # SHHS loader
│   │   ├── apnea_ecg_dataset.py    # PhysioNet Apnea-ECG
│   │   ├── mesa_sleep_dataset.py   # MESA Sleep (optional)
│   │   └── transforms.py           # Apnea-specific transforms
│   ├── models/
│   │   ├── __init__.py
│   │   ├── custom_cnn.py           # Custom CNN baseline
│   │   ├── resnet18_transfer.py    # ResNet18 TL
│   │   ├── vit_bilstm.py           # Hybrid ViT+BiLSTM
│   │   └── ssl_pretrainer.py       # Self-supervised pretraining
│   ├── training/
│   │   ├── __init__.py
│   │   ├── apnea_trainer.py        # Apnea-specific trainer
│   │   └── ssl_trainer.py          # SSL pretraining loop
│   └── utils/
│       ├── __init__.py
│       ├── ahi_computation.py      # AHI calculation
│       └── severity_labels.py      # Severity classification
│
├── src/                            # EXISTING: BOAS sleep staging
│   └── ...                         # (unchanged)
│
└── pipeline.py                     # EXISTING: BOAS pipeline (unchanged)
```

**Milestones:**
- [ ] Create directory structure
- [ ] Initialize `__init__.py` files
- [ ] Create base configuration files
- [ ] Write `sleep_apnea/README.md` with quickstart

**Dependencies:** None (pure scaffolding)

---

### Phase 2: Dataset Integration (Days 2-4)

**Goal:** Implement data loaders for SHHS and secondary dataset.

#### 2.1 SHHS Dataset Loader (`sleep_apnea/data/shhs_dataset.py`)

**SHHS Overview:**
- **Source:** [NSRR (National Sleep Research Resource)](https://sleepdata.org/datasets/shhs)
- **Size:** ~5,800 subjects with PSG recordings
- **Format:** EDF files with annotations
- **Labels:** AHI (Apnea-Hypopnea Index) for severity classification

**Severity Classification (4-class):**
```python
SEVERITY_LABELS = {
    'Healthy':    AHI < 5,      # No sleep apnea
    'Mild':       5 <= AHI < 15,
    'Moderate':   15 <= AHI < 30,
    'Severe':     AHI >= 30,
}
```

**Implementation Tasks:**
- [ ] Download script for SHHS (NSRR authentication required)
- [ ] EDF file parser with PSG channel extraction
- [ ] AHI label extraction from annotations
- [ ] Train/val/test split (subject-level, stratified by severity)
- [ ] Integration with existing scalogram pipeline (optional)

**Key Design:**
```python
class SHHSDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        channels: List[str] = ['C3-A2', 'C4-A1', 'EOG', 'EMG'],
        target_sfreq: float = 125.0,
        sequence_length: int = 3000,  # 30 seconds at 100 Hz
        aggregation: str = 'mean',    # 'mean', 'attention', 'lstm'
    ):
        ...
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        # Returns: (signal, severity_label)
        # signal: (channels, time) for single epoch
        # OR: (num_epochs, channels, time) for sequence
        ...
```

#### 2.2 Secondary Dataset: PhysioNet Apnea-ECG

**Why Apnea-ECG:**
- **Public:** No authentication required
- **Binary labels:** Apnea vs. Normal (can be aggregated)
- **ECG-based:** Complementary modality to EEG
- **Well-established:** Benchmark dataset

**Implementation:**
- [ ] Reuse existing `src/data/apnea_dataset.py` (already exists!)
- [ ] Add severity estimation from apnea annotations
- [ ] Cross-dataset evaluation protocol

#### 2.3 Alternative: MESA Sleep Dataset

**Why MESA:**
- **Diverse demographics:** Multi-ethnic cohort
- **Large sample:** ~2,200 subjects
- **Rich annotations:** AHI, oxygen desaturation, sleep stages

**Decision:** Implement SHHS first, add MESA as optional extension.

**Milestones:**
- [ ] SHHS loader with AHI extraction
- [ ] Severity label computation
- [ ] Stratified split implementation
- [ ] PhysioNet Apnea-ECG integration
- [ ] Data verification scripts

---

### Phase 3: Model Architectures (Days 5-10)

**Goal:** Implement three model architectures with self-supervised pretraining.

#### 3.1 Custom CNN Baseline (`sleep_apnea/models/custom_cnn.py`)

**Architecture:**
```
Input: (batch, channels=6, height=224, width=224) [scalograms]
       OR (batch, channels=6, time=3000) [raw EEG]

Conv Block 1: Conv2d(6, 32, 3) → BN → ReLU → MaxPool(2)
Conv Block 2: Conv2d(32, 64, 3) → BN → ReLU → MaxPool(2)
Conv Block 3: Conv2d(64, 128, 3) → BN → ReLU → MaxPool(2)
Conv Block 4: Conv2d(128, 256, 3) → BN → ReLU → GlobalAvgPool

FC Head: Linear(256, 128) → ReLU → Dropout(0.5) → Linear(128, 4)

Output: 4-class logits (Healthy, Mild, Moderate, Severe)
```

**Features:**
- [ ] Simple, interpretable baseline
- [ ] ~1-2M parameters
- [ ] Fast training (~1 hour on A100)
- [ ] Optional attention module

#### 3.2 ResNet18 Transfer Learning (`sleep_apnea/models/resnet18_transfer.py`)

**Architecture:**
```
Backbone: ResNet18 (ImageNet pretrained)
  - Modified first layer: 6-channel input (vs. 3-channel RGB)
  - Frozen early layers (optional fine-tuning strategy)

Head:
  - AdaptiveAvgPool2d(1)
  - Linear(512, 256) → ReLU → Dropout(0.5)
  - Linear(256, 4)

Output: 4-class logits
```

**Transfer Learning Strategy:**
```python
# Phase 1: Freeze backbone, train head (5 epochs)
for param in model.backbone.parameters():
    param.requires_grad = False

# Phase 2: Unfreeze last 2 layers, fine-tune (10 epochs)
for param in model.backbone.layer4.parameters():
    param.requires_grad = True

# Phase 3: Full fine-tuning (optional, 5 epochs)
for param in model.backbone.parameters():
    param.requires_grad = True
```

**Features:**
- [ ] Leverages ImageNet features
- [ ] Progressive unfreezing strategy
- [ ] Comparison point for transfer learning efficacy

#### 3.3 Hybrid ViT + BiLSTM (`sleep_apnea/models/vit_bilstm.py`)

**Core Idea:** ViT extracts spatial features from scalograms; BiLSTM models temporal dynamics across epochs.

**Architecture:**
```
Per-Epoch Encoding (ViT):
  Input: (batch, num_epochs, 3, 224, 224)
  ViT-Small: (batch, num_epochs, 384) [CLS token embeddings]

Temporal Modeling (BiLSTM):
  BiLSTM(input_size=384, hidden_size=256, num_layers=2, bidirectional=True)
  Output: (batch, 512) [concatenated final hidden states]

Classification Head:
  Linear(512, 256) → LayerNorm → ReLU → Dropout(0.5)
  Linear(256, 4)

Output: 4-class logits
```

**Implementation:**
```python
class ViTBiLSTMHybrid(nn.Module):
    def __init__(
        self,
        vit_name: str = 'vit_small_patch16_224',
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        num_classes: int = 4,
        pretrained_vit: bool = True,
    ):
        super().__init__()
        
        # ViT encoder (per-epoch)
        self.vit = timm.create_model(vit_name, pretrained=pretrained_vit, num_classes=0)
        vit_dim = self.vit.num_features
        
        # Temporal modeling
        self.lstm = nn.LSTM(
            input_size=vit_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        # x: (batch, num_epochs, 3, 224, 224)
        B, T, C, H, W = x.shape
        
        # ViT encoding (per epoch)
        x = x.view(B * T, C, H, W)
        vit_features = self.vit(x)  # (B*T, vit_dim)
        vit_features = vit_features.view(B, T, -1)  # (B, T, vit_dim)
        
        # LSTM temporal modeling
        lstm_out, (h_n, c_n) = self.lstm(vit_features)
        # Concatenate final forward and backward hidden states
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2*hidden)
        
        # Classification
        return self.classifier(final_hidden)
```

**Features:**
- [ ] **Main contribution model** for paper
- [ ] Captures both spatial (ViT) and temporal (LSTM) patterns
- [ ] ~20-30M parameters
- [ ] Expected training time: ~4-6 hours on A100

#### 3.4 Self-Supervised Pretraining (`sleep_apnea/models/ssl_pretrainer.py`)

**Method: Masked Autoencoding (MAE)**

**Why MAE:**
- Proven effective for EEG (see recent literature)
- Matches ViT architecture naturally
- No negative samples needed (simpler than SimCLR/BYOL)

**Pretraining Strategy:**
```
1. Mask 75% of input patches (random masking)
2. Encoder (ViT) processes visible patches only
3. Lightweight decoder reconstructs masked patches
4. Reconstruction loss: MSE between original and reconstructed

Fine-tuning:
1. Remove decoder
2. Add classification head
3. Fine-tune on labeled data
```

**Implementation:**
```python
class MAEPretrainer(nn.Module):
    def __init__(
        self,
        vit_name: str = 'vit_small_patch16_224',
        mask_ratio: float = 0.75,
        decoder_depth: int = 4,
        decoder_dim: int = 256,
    ):
        super().__init__()
        
        self.vit = timm.create_model(vit_name, pretrained=False, num_classes=0)
        self.patch_size = 16  # ViT-Small patch size
        self.mask_ratio = mask_ratio
        
        # Lightweight decoder
        self.decoder = nn.Sequential(
            nn.Linear(384, decoder_dim),
            nn.ReLU(),
            *[nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=8) 
              for _ in range(decoder_depth)],
            nn.Linear(decoder_dim, self.patch_size ** 2 * 3),  # Reconstruct pixels
        )
    
    def forward(self, x):
        # x: (batch, 3, 224, 224)
        # Returns: reconstruction loss
        ...
    
    def finetune_classifier(self, num_classes: int = 4) -> nn.Module:
        # Remove decoder, add classification head
        ...
```

**Pretraining Protocol:**
```yaml
# ssl_pretrain_config.yaml
pretraining:
  epochs: 100
  batch_size: 128
  learning_rate: 1.5e-4
  warmup_epochs: 10
  mask_ratio: 0.75
  
finetuning:
  epochs: 30
  batch_size: 64
  learning_rate: 1e-4
  layer_decay: 0.75  # Lower LR for early ViT layers
```

**Milestones:**
- [ ] Custom CNN implementation
- [ ] ResNet18 transfer learning with progressive unfreezing
- [ ] ViT+BiLSTM hybrid architecture
- [ ] MAE pretraining implementation
- [ ] Fine-tuning pipeline for pretrained models
- [ ] Model verification tests

---

### Phase 4: Training Pipeline (Days 11-14)

**Goal:** Implement training infrastructure for apnea classification.

#### 4.1 Apnea-Specific Trainer (`sleep_apnea/training/apnea_trainer.py`)

**Key Differences from BOAS Trainer:**
- 4-class classification (vs. 5-class sleep staging)
- Class imbalance handling (severity distribution skewed)
- Sequence modeling support (for ViT+BiLSTM)
- Cross-dataset evaluation

**Implementation:**
```python
class ApneaTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: ApneaConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
    ):
        ...
    
    def train_epoch(self) -> Dict[str, float]:
        # Standard training loop with mixed precision
        ...
    
    def validate(self) -> Dict[str, float]:
        # Validation with comprehensive metrics
        ...
    
    def fit(self) -> Dict[str, Any]:
        # Full training loop with early stopping
        ...
```

**Class Imbalance Handling:**
```python
# Compute class weights from training distribution
class_weights = 1.0 / (class_counts / class_counts.sum())
criterion = nn.CrossEntropyLoss(weight=class_weights)

# OR use focal loss for hard examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        ...
```

#### 4.2 SSL Pretraining Loop (`sleep_apnea/training/ssl_trainer.py`)

```python
class SSLTrainer:
    def __init__(
        self,
        model: MAEPretrainer,
        config: SSLConfig,
        train_loader: DataLoader,  # Unlabeled data
        device: torch.device,
    ):
        ...
    
    def train_epoch(self) -> Dict[str, float]:
        # MAE training: reconstruction loss
        ...
    
    def save_pretrained(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, path)
```

#### 4.3 Cross-Dataset Evaluation Protocol

**Evaluation Scenarios:**
```
1. In-domain: Train on SHHS, Test on SHHS (held-out subjects)
2. Cross-dataset: Train on SHHS, Test on Apnea-ECG
3. Cross-dataset: Train on Apnea-ECG, Test on SHHS
4. Multi-task: Train on both, Test on each
```

**Implementation:**
```python
def evaluate_cross_dataset(
    model: nn.Module,
    source_dataset: str,
    target_dataset: str,
) -> Dict[str, float]:
    """Evaluate generalization across datasets."""
    ...
```

**Milestones:**
- [ ] Apnea trainer with class imbalance handling
- [ ] SSL pretraining loop
- [ ] Fine-tuning pipeline
- [ ] Cross-dataset evaluation protocol
- [ ] Checkpointing and resume functionality

---

### Phase 5: Integration & CLI (Days 15-17)

**Goal:** Create unified command-line interface for running experiments.

#### 5.1 Main Entry Point (`sleep_apnea/run_apnea.py`)

```python
#!/usr/bin/env python3
"""
Sleep Apnea Severity Classification Pipeline.

Usage:
    # Train Custom CNN baseline
    python run_apnea.py --model cnn_baseline --epochs 30
    
    # Train ResNet18 with transfer learning
    python run_apnea.py --model resnet18_transfer --epochs 30 --pretrained
    
    # Train ViT+BiLSTM with SSL pretraining
    python run_apnea.py --model vit_bilstm --epochs 30 --ssl_pretrain
    
    # Cross-dataset evaluation
    python run_apnea.py --model vit_bilstm --checkpoint path/to/ckpt.pt --eval-only --target-dataset apnea_ecg
"""

import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Sleep Apnea Classification')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True,
                       choices=['cnn_baseline', 'resnet18_transfer', 'vit_bilstm'],
                       help='Model architecture to train')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='shhs',
                       choices=['shhs', 'apnea_ecg', 'mesa'],
                       help='Dataset to use')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    # SSL pretraining
    parser.add_argument('--ssl-pretrain', action='store_true',
                       help='Use self-supervised pretraining')
    parser.add_argument('--ssl-epochs', type=int, default=100,
                       help='SSL pretraining epochs')
    
    # Evaluation
    parser.add_argument('--eval-only', action='store_true',
                       help='Skip training, only evaluate')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint for evaluation')
    parser.add_argument('--target-dataset', type=str,
                       help='Target dataset for cross-dataset evaluation')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./sleep_apnea/outputs',
                       help='Output directory for results')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name for logging')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='DataLoader workers')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    seed_everything(args.seed)
    
    # Load config
    config = load_config(args.model)
    
    # Load dataset
    train_loader, val_loader, test_loader = load_dataset(
        args.dataset, args.data_dir, args.batch_size, args.num_workers
    )
    
    # Create model
    model = create_model(args.model, config)
    
    # SSL pretraining (optional)
    if args.ssl_pretrain:
        print("Starting SSL pretraining...")
        ssl_model = pretrain_ssl(model, train_loader, args)
        model = ssl_model.finetune_classifier(num_classes=4)
    
    # Training
    if not args.eval_only:
        trainer = ApneaTrainer(model, config, train_loader, val_loader, test_loader)
        results = trainer.fit()
    
    # Evaluation
    if args.eval_only or args.target_dataset:
        evaluate_model(model, test_loader, args)
    
    print("Experiment complete!")


if __name__ == '__main__':
    main()
```

#### 5.2 Configuration Files

**`sleep_apnea/configs/default_config.yaml`:**
```yaml
# Default configuration for sleep apnea classification

# Dataset
dataset:
  name: shhs
  channels: ['C3-A2', 'C4-A1', 'EOG-L', 'EOG-R', 'EMG', 'Resp']
  target_sfreq: 125.0
  sequence_length: 3000  # 30 seconds
  
# Model
model:
  num_classes: 4
  class_names: ['Healthy', 'Mild', 'Moderate', 'Severe']
  
# Training
training:
  epochs: 30
  batch_size: 64
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  optimizer: adamw
  scheduler: cosine
  warmup_epochs: 5
  
# Class imbalance
class_weights: true
focal_loss:
  enabled: false
  alpha: 1.0
  gamma: 2.0
  
# SSL pretraining
ssl:
  enabled: false
  method: mae
  epochs: 100
  mask_ratio: 0.75
  pretrain_lr: 1.5e-4
  
# Evaluation
evaluation:
  metrics: ['accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall', 'auc_roc', 'cohens_kappa']
  cross_dataset: false
  
# Output
output:
  dir: ./sleep_apnea/outputs
  log_dir: ./sleep_apnea/logs
  checkpoint_dir: ./sleep_apnea/checkpoints
```

**Milestones:**
- [ ] Main entry point with CLI
- [ ] Configuration file system
- [ ] Experiment logging (TensorBoard, W&B optional)
- [ ] Checkpoint management
- [ ] Results aggregation script

---

### Phase 6: Experiments & Validation (Days 18-21)

**Goal:** Run comprehensive experiments and validate results.

#### 6.1 Experiment Matrix

| Experiment | Model | SSL | Dataset | Expected Acc | Priority |
|------------|-------|-----|---------|--------------|----------|
| E1 | CNN Baseline | No | SHHS | 65-70% | High |
| E2 | CNN Baseline | Yes | SHHS | 68-73% | Medium |
| E3 | ResNet18 TL | No | SHHS | 72-77% | High |
| E4 | ResNet18 TL | Yes | SHHS | 75-80% | Medium |
| E5 | ViT+BiLSTM | No | SHHS | 75-80% | High |
| E6 | ViT+BiLSTM | Yes | SHHS | **78-83%** | **Critical** |
| E7 | ViT+BiLSTM | Yes | SHHS→Apnea-ECG | 70-75% | High |
| E8 | ViT+BiLSTM | Yes | Apnea-ECG→SHHS | 72-77% | High |

#### 6.2 Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| **Accuracy (4-class)** | >75% | Main metric |
| **F1-Macro** | >70% | Handles class imbalance |
| **Cohen's Kappa** | >0.60 | Agreement beyond chance |
| **Cross-dataset drop** | <10% | Generalization measure |
| **SSL improvement** | >3% | Pretraining efficacy |

#### 6.3 Ablation Studies

1. **Architecture ablation:**
   - ViT only (no LSTM)
   - LSTM only (no ViT)
   - ViT+BiLSTM (full model)

2. **SSL ablation:**
   - No pretraining
   - MAE pretraining
   - SimCLR pretraining (optional)

3. **Input modality:**
   - EEG only
   - EEG + EOG
   - Full PSG (EEG + EOG + EMG + Resp)

**Milestones:**
- [ ] Run all baseline experiments (E1-E5)
- [ ] Run main model experiments (E6)
- [ ] Cross-dataset evaluation (E7-E8)
- [ ] Ablation studies
- [ ] Results aggregation and analysis

---

## ⚠️ Risks & Bottlenecks

### Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **SHHS access delays** | High | Medium | Start NSRR application early; use Apnea-ECG as fallback |
| **Class imbalance** | High | High | Use class weights, focal loss, oversampling |
| **SSL pretraining instability** | Medium | Medium | Use proven MAE hyperparameters; gradient clipping |
| **Cross-dataset domain shift** | High | High | Domain adaptation techniques; data augmentation |
| **GPU memory (ViT+BiLSTM)** | Medium | Medium | Gradient accumulation; mixed precision; smaller batches |

### Timeline Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Dataset preprocessing takes longer than expected | High | Start with small subset; parallelize preprocessing |
| Model convergence issues | Medium | Extensive hyperparameter tuning; use proven baselines |
| Cross-dataset evaluation fails | Medium | Have fallback evaluation protocol |

### Bottlenecks

1. **Dataset Download:** SHHS requires NSRR approval (1-2 weeks)
   - **Mitigation:** Start application immediately; use Apnea-ECG for development

2. **SSL Pretraining Time:** 100 epochs on full SHHS (~12 hours)
   - **Mitigation:** Use subset for hyperparameter tuning; full dataset for final pretraining

3. **ViT+BiLSTM Training:** Sequence modeling is memory-intensive
   - **Mitigation:** Gradient accumulation; mixed precision; gradient checkpointing

---

## 🚀 Next Immediate Actions

### Right Now (Today)

1. **Create directory structure:**
   ```bash
   cd D:\Projects\AI-Projects\EEG
   mkdir -p sleep_apnea/{configs,data,models,training,utils,outputs,logs,checkpoints}
   touch sleep_apnea/__init__.py
   touch sleep_apnea/{data,models,training,utils}/__init__.py
   ```

2. **Start NSRR application for SHHS access:**
   - Visit: https://sleepdata.org/datasets/shhs
   - Complete data use agreement
   - Expected approval: 1-2 weeks

3. **Create base configuration file:**
   - Copy `sleep_apnea/configs/default_config.yaml` from this plan

4. **Implement SHHS dataset loader skeleton:**
   - Start with `sleep_apnea/data/shhs_dataset.py`
   - Use existing `src/data/boas_dataset.py` as template

### This Week

1. Complete Phase 1-2 (structure + dataset loader)
2. Test with small subset of data
3. Implement Custom CNN baseline
4. Verify training pipeline works end-to-end

### Next Week

1. Implement ResNet18 and ViT+BiLSTM
2. Implement SSL pretraining
3. Start baseline experiments

---

## 📚 Appendix

### A. Dataset Summary

| Dataset | Subjects | Modality | Labels | Access |
|---------|----------|----------|--------|--------|
| **SHHS** | ~5,800 | PSG (EEG, EOG, EMG, Resp) | AHI (continuous) | NSRR application |
| **Apnea-ECG** | 70 | ECG (single lead) | Binary (apnea/normal) | Public (PhysioNet) |
| **MESA Sleep** | ~2,200 | PSG (full) | AHI, sleep stages | dbGaP application |

### B. Recommended Hyperparameters

```yaml
# ViT+BiLSTM (main model)
model:
  vit_name: vit_small_patch16_224
  lstm_hidden: 256
  lstm_layers: 2
  dropout: 0.3

training:
  epochs: 30
  batch_size: 32  # Reduced for sequence modeling
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  gradient_clip: 1.0
  
ssl:
  epochs: 100
  mask_ratio: 0.75
  learning_rate: 1.5e-4
```

### C. Command Quick Reference

```bash
# Train CNN baseline
python sleep_apnea/run_apnea.py --model cnn_baseline --data-dir /path/to/shhs --epochs 30

# Train ViT+BiLSTM with SSL
python sleep_apnea/run_apnea.py --model vit_bilstm --data-dir /path/to/shhs --epochs 30 --ssl-pretrain

# Cross-dataset evaluation
python sleep_apnea/run_apnea.py --model vit_bilstm --checkpoint checkpoints/vit_bilstm_best.pt --eval-only --target-dataset apnea_ecg
```

---

**Document Status:** ✅ Complete and Ready for Implementation  
**Estimated Total Effort:** 3-4 weeks (full implementation)  
**Critical Path:** SHHS access → Dataset loader → ViT+BiLSTM → SSL pretraining → Experiments
