# 🫁 Sleep Apnea Severity Classification Pipeline

**Status:** 🚧 Under Development  
**Target:** 4-class sleep apnea severity classification (Healthy/Mild/Moderate/Severe)

---

## 🎯 Overview

This pipeline implements sleep apnea severity classification using multiple deep learning architectures with self-supervised pretraining.

### Key Features

- **4-Class Classification:** Healthy (AHI<5), Mild (5-15), Moderate (15-30), Severe (30+)
- **Multiple Architectures:**
  - Custom CNN (baseline)
  - ResNet18 Transfer Learning
  - Hybrid ViT + BiLSTM (main model)
- **Self-Supervised Pretraining:** MAE (Masked Autoencoding)
- **Cross-Dataset Validation:** SHHS → Apnea-ECG generalization testing

---

## 🚀 Quick Start

### Prerequisites

1. **SHHS Dataset Access:** Apply at https://sleepdata.org/datasets/shhs
2. **Python 3.10+** with dependencies from root `requirements.txt`

### Installation

```bash
# From project root
cd D:\Projects\AI-Projects\EEG
pip install -r requirements.txt
```

### Basic Usage

```bash
# Train CNN baseline
python -m sleep_apnea.run_apnea --model cnn_baseline --data-dir /path/to/shhs --epochs 30

# Train ViT+BiLSTM with SSL pretraining
python -m sleep_apnea.run_apnea --model vit_bilstm --data-dir /path/to/shhs --epochs 30 --ssl-pretrain

# Evaluate checkpoint
python -m sleep_apnea.run_apnea --model vit_bilstm --checkpoint checkpoints/best.pt --eval-only
```

---

## 📊 Dataset

### Primary: SHHS (Sleep Heart Health Study)

| Property | Value |
|----------|-------|
| Subjects | ~5,800 |
| Modality | Full PSG (EEG, EOG, EMG, Resp) |
| Labels | AHI (Apnea-Hypopnea Index) |
| Access | NSRR application required |

### Secondary: PhysioNet Apnea-ECG

| Property | Value |
|----------|-------|
| Subjects | 70 |
| Modality | Single-lead ECG |
| Labels | Binary (apnea/normal per minute) |
| Access | Public (PhysioNet) |

---

## 🏗️ Architecture

```
sleep_apnea/
├── configs/              # YAML configuration files
├── data/                 # Dataset loaders
│   ├── shhs_dataset.py
│   ├── apnea_ecg_dataset.py
│   └── transforms.py
├── models/               # Model architectures
│   ├── custom_cnn.py
│   ├── resnet18_transfer.py
│   ├── vit_bilstm.py
│   └── ssl_pretrainer.py
├── training/             # Training loops
│   ├── apnea_trainer.py
│   └── ssl_trainer.py
├── utils/                # Utilities
│   ├── ahi_computation.py
│   └── severity_labels.py
├── outputs/              # Experiment results
├── logs/                 # Training logs
├── checkpoints/          # Model checkpoints
└── run_apnea.py          # Main entry point
```

---

## 📈 Expected Results

| Model | SSL | Expected Accuracy | Training Time |
|-------|-----|-------------------|---------------|
| CNN Baseline | No | 65-70% | ~1 hour |
| CNN Baseline | Yes | 68-73% | ~3 hours |
| ResNet18 TL | No | 72-77% | ~2 hours |
| ResNet18 TL | Yes | 75-80% | ~4 hours |
| ViT+BiLSTM | No | 75-80% | ~4 hours |
| **ViT+BiLSTM** | **Yes** | **78-83%** | **~8 hours** |

---

## 📝 Configuration

Edit `configs/default_config.yaml` for global settings, or use CLI flags:

```yaml
# Example config override
training:
  epochs: 50
  batch_size: 32
  learning_rate: 1.0e-4
  
ssl:
  enabled: true
  epochs: 100
  mask_ratio: 0.75
```

---

## 🔬 Experiment Tracking

Results are saved to `outputs/` with the following structure:

```
outputs/
├── cnn_baseline/
│   ├── run_001/
│   │   ├── checkpoint_best.pt
│   │   ├── results.json
│   │   └── logs/
│   └── run_002/
├── resnet18_transfer/
└── vit_bilstm/
```

---

## 📋 TODO

- [ ] SHHS dataset loader
- [ ] Custom CNN baseline
- [ ] ResNet18 transfer learning
- [ ] ViT+BiLSTM hybrid
- [ ] MAE pretraining
- [ ] Cross-dataset evaluation
- [ ] Comprehensive experiments

---

## 📄 License

Same as parent project.

---

**See:** `../../SLEEP_APNEA_CLASSIFICATION_PLAN.md` for full implementation plan.
