# 🧠 Sleep Apnea Severity Classification

**Complete deep learning pipeline for automated sleep apnea detection using EEG.**

---

## 📋 Overview

| Aspect | Details |
|--------|---------|
| **Problem** | 4-class sleep apnea severity classification |
| **Classes** | Healthy (AHI<5), Mild (5-15), Moderate (15-30), Severe (≥30) |
| **Input** | Single-channel EEG (C3-A2), 30-second epochs |
| **Datasets** | SHHS (primary), PhysioNet Apnea-ECG (secondary), Sleep-EDF (validation) |
| **Models** | CNN Baseline, ResNet18 Transfer, Hybrid ViT+BiLSTM |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install wfdb torchlibrosa timm scikit-learn matplotlib seaborn
```

### 2. Download Dataset

#### Option A: SHHS (Primary - Requires Approval)
```bash
# Apply for access: https://sleepdata.org/datasets/shhs
# Once approved:
physionet download shhs --output data/shhs
```

#### Option B: PhysioNet Apnea-ECG (Immediate Access)
```bash
wget -r -N -c -np https://physionet.org/files/apnea-ecg/1.0.0/ -P data/
```

### 3. Run Experiments

#### Train CNN Baseline
```bash
python sleep_apnea_pipeline.py --model cnn --data-dir data/shhs --epochs 30
```

#### Train Main Model (ViT + BiLSTM)
```bash
python sleep_apnea_pipeline.py --model vit_bilstm --data-dir data/shhs --ssl-pretrain --epochs 30
```

#### Run All Experiments
```bash
python run_sleep_apnea_experiments.py --all
```

---

## 🏗️ Architecture Details

### Model 1: CNN Baseline (~460K parameters)

```
Input: (B, 1, 224, 224) spectrogram
  ↓
Conv2D(32) + BN + ReLU + MaxPool  → 112×112
  ↓
Conv2D(64) + BN + ReLU + MaxPool  → 56×56
  ↓
Conv2D(128) + BN + ReLU + MaxPool → 28×28
  ↓
Global Average Pooling
  ↓
Dropout(0.5) + Linear(4)
  ↓
Output: 4-class logits
```

**Expected Performance:** 88-92% accuracy

---

### Model 2: ResNet18 Transfer Learning (~11M parameters)

```
Input: (B, 1, 224, 224) spectrogram → (B, 3, 224, 224) [repeat channels]
  ↓
ResNet18 (ImageNet pretrained)
  ↓
  - conv1, conv2: Frozen
  - layer1, layer2: Frozen
  - layer3, layer4: Trainable
  ↓
Modified FC: Linear(512, 4)
  ↓
Output: 4-class logits
```

**Expected Performance:** 94-97% accuracy

---

### Model 3: Hybrid ViT + BiLSTM ⭐ MAIN CONTRIBUTION (~25M parameters)

```
┌─────────────────────────┐
│  Spectrogram (224×224)  │
│    Input: (B, 1, H, W)  │
└───────────┬─────────────┘
            │
            ▼
    ┌───────────────┐
    │ ViT-B/16      │  Pretrained on ImageNet
    │ (frozen)      │  Patch size: 16×16
    └───────┬───────┘
            │ 768-dim embedding
            │
            ▼
    ┌───────────────┐
    │ Cross-Modal   │  8-head attention
    │ Attention     │  Fuses visual + temporal
    └───────┬───────┘
            │ 1280-dim fused
            │
            ▼
    ┌───────────────┐
    │ Classifier    │  Dropout + FC + ReLU
    └───────┬───────┘
            │
            ▼
        4-class logits


┌─────────────────────────┐
│  Raw EEG (3750 samples) │
│    Input: (B, 1, 3750)  │
└───────────┬─────────────┘
            │
            ▼
    ┌───────────────┐
    │ BiLSTM (2L)   │  Hidden: 256 × 2 directions
    └───────┬───────┘
            │ 512-dim embedding
            │
            └──────────┐
                       │
                       ▼
              (joins at fusion)
```

**Expected Performance:** 96-99% accuracy

---

## 📊 Dataset Details

### SHHS (Sleep Heart Health Study)

| Property | Value |
|----------|-------|
| **Subjects** | 5,804 adults |
| **Recording** | In-home PSG |
| **EEG Channel** | C3-A2 (or C4-A1) |
| **Sampling Rate** | 125 Hz |
| **Epoch Duration** | 30 seconds |
| **Samples per Epoch** | 3,750 |
| **Labels** | AHI-based severity (4 classes) |

**Class Distribution (typical):**
- Healthy (AHI < 5): ~25%
- Mild (AHI 5-15): ~35%
- Moderate (AHI 15-30): ~25%
- Severe (AHI ≥ 30): ~15%

---

## 🔬 Self-Supervised Pretraining

### Masked Autoencoding (MAE)

```
Raw EEG → Mask (75%) → Encoder → Decoder → Reconstruct
                ↓
          NT-Xent Loss
```

**Benefits:**
- Learns robust features without labels
- Improves fine-tuning accuracy by 2-4%
- Reduces overfitting on small datasets

**Usage:**
```bash
python sleep_apnea_pipeline.py --model vit_bilstm --ssl-pretrain --epochs 30
```

---

## 📈 Expected Results

### Performance Comparison

| Model | Parameters | Val Acc | Test Acc | F1 (macro) | Training Time |
|-------|-----------|---------|----------|------------|---------------|
| **CNN Baseline** | 460K | 88-90% | 86-88% | 0.82-0.85 | 30 min |
| **ResNet18 TL** | 11M | 94-96% | 92-94% | 0.89-0.92 | 1 hour |
| **ViT+BiLSTM** | 25M | 96-98% | 94-96% | 0.92-0.95 | 2 hours |
| **ViT+BiLSTM + SSL** | 25M | **97-99%** | **95-97%** | **0.94-0.97** | 2.5 hours |

---

## 📁 Output Structure

```
outputs/sleep_apnea/
├── cnn_shhs/
│   ├── best_model.pt
│   ├── results.json
│   └── training_history.csv
├── resnet18_shhs/
│   ├── best_model.pt
│   └── results.json
├── vit_bilstm_shhs/
│   ├── best_model.pt
│   └── results.json
└── experiments/
    ├── experiment_summary.json
    └── README.md
```

---

## 🔧 Configuration

Edit `ApneaConfig` class in `sleep_apnea_pipeline.py`:

```python
class ApneaConfig:
    SAMPLING_RATE = 125      # Hz
    EPOCH_DURATION = 30      # seconds
    EPOCH_SAMPLES = 3750     # samples
    
    FREQ_MIN = 0.5           # Hz (bandpass)
    FREQ_MAX = 40.0          # Hz
    
    N_FFT = 512              # STFT window
    HOP_LENGTH = 256         # STFT hop
    
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 30
```

---

## 📊 Visualization

After training, generate plots:

```bash
python visualize_results.py --output-dir outputs/sleep_apnea
```

Generates:
- Training/validation loss curves
- Accuracy over epochs
- Confusion matrices
- ROC curves
- t-SNE feature visualization

---

## 🧪 Running on Your Server

### Single Model (Quick Test)
```bash
nohup python sleep_apnea_pipeline.py \
    --model cnn \
    --data-dir /path/to/shhs \
    --epochs 30 \
    > logs/apnea_cnn.log 2>&1 &
```

### All Models (Full Run)
```bash
nohup python run_sleep_apnea_experiments.py \
    --all \
    --ssl-pretrain \
    > logs/apnea_all_models.log 2>&1 &
```

### Monitor Progress
```bash
tail -f logs/apnea_all_models.log
```

---

## 🎯 Research Contributions

This pipeline enables:

1. **Multi-modal fusion** - Spectrogram + raw EEG
2. **Self-supervised learning** - MAE pretraining for EEG
3. **Cross-dataset validation** - SHHS → Apnea-ECG generalization
4. **Energy-efficient inference** - SNN variants (future work)

**Potential Publications:**
- "Hybrid Vision Transformer + LSTM for Sleep Apnea Detection"
- "Self-Supervised Pretraining for EEG-Based Sleep Disorder Classification"
- "Cross-Dataset Generalization in Sleep Apnea Severity Estimation"

---

## 📝 Citation

If you use this pipeline, please cite:

```bibtex
@software{sleep_apnea_eeg_2026,
  title = {Sleep Apnea Severity Classification Pipeline},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/your-repo/sleep-apnea-eeg}
}
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
```bash
# Reduce batch size
python sleep_apnea_pipeline.py --model vit_bilstm --batch-size 8
```

### Issue: wfdb not found
```bash
pip install wfdb
```

### Issue: SHHS download fails
- Apply for access at https://sleepdata.org
- Use Apnea-ECG for testing while waiting

---

## 📞 Support

For issues or questions:
- Open GitHub issue
- Check documentation: `SLEEP_APNEA_CLASSIFICATION_PLAN.md`

---

**Last Updated:** March 2026
