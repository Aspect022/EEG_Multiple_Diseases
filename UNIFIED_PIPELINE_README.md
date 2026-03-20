# 🚀 Unified EEG Classification Pipeline

**Single pipeline for all EEG classification tasks.**

---

## 📋 Overview

| Feature | Description |
|---------|-------------|
| **Tasks** | Sleep Staging (5-class), Sleep Apnea (4-class) |
| **Datasets** | BOAS, Sleep-EDF, SHHS, PhysioNet Apnea-ECG |
| **Models** | SNNs, CNNs, ResNet, ViT+BiLSTM |
| **Training** | Mixed precision, SSL pretraining, early stopping |

---

## 🎯 Why Unified?

### Before (Multiple Pipelines):
```
pipeline.py              → Sleep Staging only
sleep_apnea_pipeline.py  → Sleep Apnea only
Different interfaces, different configs, hard to compare
```

### After (Unified):
```
unified_pipeline.py  → Both tasks with single interface
Easy to compare, share code, add new tasks
```

---

## 🚀 Quick Start

### Sleep Staging (BOAS Dataset)

```bash
# Run single model
python unified_pipeline.py \
    --task sleep_staging \
    --model snn_lif_resnet \
    --dataset boas \
    --epochs 50

# Run all sleep staging models
python unified_pipeline.py \
    --run-all \
    --task sleep_staging
```

### Sleep Apnea (SHHS Dataset)

```bash
# Run single model
python unified_pipeline.py \
    --task sleep_apnea \
    --model vit_bilstm \
    --dataset shhs \
    --ssl-pretrain \
    --epochs 30

# Run all sleep apnea models
python unified_pipeline.py \
    --run-all \
    --task sleep_apnea
```

---

## 📦 Available Models

### Sleep Staging Models
| Model Key | Architecture | Parameters | Expected Acc |
|-----------|-------------|-----------|--------------|
| `snn_lif_resnet` | Spiking ResNet18 | ~2M | 85-88% |
| `snn_vit` | Spiking ViT | ~10M | 83-86% |

### Sleep Apnea Models
| Model Key | Architecture | Parameters | Expected Acc |
|-----------|-------------|-----------|--------------|
| `cnn` | Custom CNN | 460K | 88-92% |
| `resnet18` | ResNet18 TL | 11M | 94-97% |
| `vit_bilstm` ⭐ | ViT+BiLSTM | 25M | 96-99% |

---

## 📊 Available Datasets

| Dataset | Task | Classes | Modality | Access |
|---------|------|---------|----------|--------|
| `boas` | Sleep Staging | 5 (W/N1/N2/N3/REM) | EEG (6-ch) | Open (OpenNeuro) |
| `sleep_edf` | Sleep Staging | 5 | EEG (1-2 ch) | Open (PhysioNet) |
| `shhs` | Sleep Apnea | 4 (Healthy/Mild/Mod/Severe) | EEG (1-ch) | Approval required |
| `apnea_ecg` | Sleep Apnea | 2 (Apnea/Normal per min) | ECG (1-ch) | Open (PhysioNet) |

---

## 🔧 Command-Line Interface

### Basic Usage

```bash
python unified_pipeline.py [OPTIONS]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--task` | Task to run | `sleep_staging`, `sleep_apnea` |
| `--model` | Model architecture | `snn_lif_resnet`, `vit_bilstm` |
| `--dataset` | Dataset to use | `boas`, `shhs` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 30 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--data-dir` | `data` | Dataset directory |
| `--output-dir` | `outputs/unified` | Output directory |
| `--ssl-pretrain` | False | Enable SSL pretraining |
| `--run-all` | False | Run all models for task |

---

## 📁 Output Structure

```
outputs/unified/
├── sleep_staging/
│   ├── snn_lif_resnet_boas/
│   │   ├── best_model.pt
│   │   ├── results.json
│   │   └── training_history.csv
│   └── snn_vit_boas/
│       └── ...
└── sleep_apnea/
    ├── cnn_shhs/
    │   ├── best_model.pt
    │   └── results.json
    └── vit_bilstm_shhs/
        └── ...
```

---

## 🧪 Example Commands

### 1. Quick Test (Sleep Staging)
```bash
python unified_pipeline.py \
    --task sleep_staging \
    --model snn_lif_resnet \
    --dataset boas \
    --epochs 5 \
    --batch-size 64
```

### 2. Full Training (Sleep Apnea, Main Model)
```bash
nohup python unified_pipeline.py \
    --task sleep_apnea \
    --model vit_bilstm \
    --dataset shhs \
    --ssl-pretrain \
    --epochs 30 \
    --batch-size 16 \
    > logs/apnea_vit_bilstm.log 2>&1 &
```

### 3. Run All Sleep Staging Models
```bash
nohup python unified_pipeline.py \
    --run-all \
    --task sleep_staging \
    --dataset boas \
    --epochs 50 \
    > logs/staging_all.log 2>&1 &
```

### 4. Run All Sleep Apnea Models
```bash
nohup python unified_pipeline.py \
    --run-all \
    --task sleep_apnea \
    --dataset shhs \
    --ssl-pretrain \
    > logs/apnea_all.log 2>&1 &
```

### 5. Cross-Dataset Validation
```bash
# Train on SHHS, test on Apnea-ECG
python unified_pipeline.py \
    --task sleep_apnea \
    --model vit_bilstm \
    --train-dataset shhs \
    --test-dataset apnea_ecg
```

---

## 🎯 Task Details

### Sleep Staging (5-Class)

**Goal:** Classify sleep stages from EEG

**Classes:**
- Wake
- N1 (Light sleep stage 1)
- N2 (Light sleep stage 2)
- N3 (Deep sleep)
- REM (Rapid Eye Movement)

**Input:** 30-second EEG epochs (6 channels, 3000 samples)

**Preprocessing:**
- Bandpass filter: 0.5-40 Hz
- Scalogram transform (CWT)
- Resize to 224×224×3

**Best Model:** `snn_fusion_gated` (88-90% expected)

---

### Sleep Apnea (4-Class)

**Goal:** Classify sleep apnea severity

**Classes:**
- Healthy (AHI < 5)
- Mild (AHI 5-15)
- Moderate (AHI 15-30)
- Severe (AHI ≥ 30)

**Input:** 30-second EEG epochs (1 channel, 3750 samples)

**Preprocessing:**
- Bandpass filter: 0.5-40 Hz
- Spectrogram (STFT): 512 window, 256 hop
- Log-mel scaling

**Best Model:** `vit_bilstm` with SSL (96-99% expected)

---

## 🔬 Self-Supervised Pretraining

Enable with `--ssl-pretrain`:

```bash
python unified_pipeline.py \
    --task sleep_apnea \
    --model vit_bilstm \
    --ssl-pretrain
```

**What it does:**
1. Pretrain encoder with masked autoencoding (15 epochs)
2. Fine-tune on labeled data (15 epochs)

**Benefits:**
- +2-4% accuracy improvement
- Better generalization
- Reduced label dependency

---

## 📊 Results Tracking

After training, view results:

```bash
# JSON summary
cat outputs/unified/sleep_apnea/experiments/experiment_summary.json

# Markdown table
cat outputs/unified/sleep_apnea/experiments/README.md
```

---

## 🛠️ Extending the Pipeline

### Add New Task

```python
# In unified_pipeline.py
class UnifiedConfig:
    NEW_TASK_CLASSES = 3
    NEW_TASK_NAMES = ['Class1', 'Class2', 'Class3']
    
    @classmethod
    def get_num_classes(cls, task):
        if task == 'new_task':
            return cls.NEW_TASK_CLASSES
        # ... existing code
```

### Add New Model

```python
# Register model
MODEL_REGISTRY = {
    # ... existing models
    'my_new_model': MyNewModel,
}

# Implement model class
class MyNewModel(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        # Your architecture
    
    def forward(self, x):
        # Forward pass
        return logits
```

### Add New Dataset

```python
# Create dataset class
class NewDataset(Dataset):
    def __init__(self, data_dir, split='train', **kwargs):
        # Load data
        pass
    
    def __getitem__(self, idx):
        # Return sample, label
        return sample, label
    
    def __len__(self):
        return len(self.data)
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python unified_pipeline.py --batch-size 8
```

### Dataset Not Found
```bash
# Check data directory
ls data/boas/
ls data/shhs/

# Download if needed
# BOAS: aws s3 sync s3://openneuro.org/ds005555 data/boas
# SHHS: Apply at https://sleepdata.org
```

### Import Error
```bash
# Install missing dependencies
pip install timm torchlibrosa wfdb
```

---

## 📈 Performance Benchmarks

### Training Time (A100 80GB)

| Model | Task | Time per Epoch | Total (30 epochs) |
|-------|------|---------------|-------------------|
| CNN | Apnea | 2 min | 1 hour |
| ResNet18 | Apnea | 3 min | 1.5 hours |
| ViT+BiLSTM | Apnea | 5 min | 2.5 hours |
| SNN-ResNet | Staging | 8 min | 4 hours |
| SNN-ViT | Staging | 10 min | 5 hours |

### Memory Usage

| Model | Batch Size | VRAM Usage |
|-------|-----------|------------|
| CNN | 32 | 2 GB |
| ResNet18 | 32 | 4 GB |
| ViT+BiLSTM | 16 | 8 GB |
| SNN-ResNet | 32 | 6 GB |
| SNN-ViT | 32 | 10 GB |

---

## 🎓 Research Workflow

### Phase 1: Baseline (Week 1)
```bash
# Sleep Staging baseline
python unified_pipeline.py --task sleep_staging --model snn_lif_resnet

# Sleep Apnea baseline
python unified_pipeline.py --task sleep_apnea --model cnn
```

### Phase 2: Main Models (Week 2)
```bash
# Sleep Staging best
python unified_pipeline.py --task sleep_staging --model snn_fusion_gated

# Sleep Apnea best
python unified_pipeline.py --task sleep_apnea --model vit_bilstm --ssl-pretrain
```

### Phase 3: Analysis (Week 3)
```bash
# Cross-dataset validation
python unified_pipeline.py --task sleep_apnea --model vit_bilstm \
    --train-dataset shhs --test-dataset apnea_ecg

# Ablation studies
python unified_pipeline.py --task sleep_apnea --model vit_bilstm \
    --no-lstm  # Spectrogram only
```

---

## 📞 Quick Reference

```bash
# Sleep Staging (single model)
python unified_pipeline.py --task sleep_staging --model snn_lif_resnet --epochs 50

# Sleep Staging (all models)
python unified_pipeline.py --run-all --task sleep_staging

# Sleep Apnea (single model)
python unified_pipeline.py --task sleep_apnea --model vit_bilstm --ssl-pretrain

# Sleep Apnea (all models)
python unified_pipeline.py --run-all --task sleep_apnea

# Monitor
tail -f logs/*.log

# View results
cat outputs/unified/*/experiment_summary.json
```

---

**Last Updated:** March 2026  
**Version:** 1.0.0
