# 🚀 Quick Start Guide - Sleep Apnea Classification

## Prerequisites

1. **Python 3.10+** with dependencies from root `requirements.txt`
2. **Dataset access:**
   - SHHS: Apply at https://sleepdata.org/datasets/shhs (1-2 weeks approval)
   - Apnea-ECG: Download from https://physionet.org/content/apnea-ecg/ (immediate)

## Installation

```bash
# From project root
cd D:\Projects\AI-Projects\EEG

# Verify dependencies
pip install -r requirements.txt

# Verify sleep_apnea module
python -c "from sleep_apnea.models import create_cnn_baseline; print('✅ Module loaded')"
```

## Quick Commands

### 1. Test with Dummy Data (No Dataset Required)

```bash
# Test CNN baseline (uses placeholder data)
python -m sleep_apnea.run_apnea \
    --model cnn_baseline \
    --data-dir ./data/dummy \
    --epochs 1 \
    --batch-size 4
```

### 2. Train with Apnea-ECG (Public Dataset)

```bash
# Download Apnea-ECG first
# Then train
python -m sleep_apnea.run_apnea \
    --model cnn_baseline \
    --dataset apnea_ecg \
    --data-dir /path/to/apnea-ecg \
    --epochs 30 \
    --batch-size 64
```

### 3. Train Main Model (ViT+BiLSTM with SSL)

```bash
# Full training with SSL pretraining
python -m sleep_apnea.run_apnea \
    --model vit_bilstm \
    --dataset shhs \
    --data-dir /path/to/shhs \
    --epochs 30 \
    --batch-size 32 \
    --ssl-pretrain \
    --ssl-epochs 100
```

### 4. Evaluate Checkpoint

```bash
python -m sleep_apnea.run_apnea \
    --model vit_bilstm \
    --dataset shhs \
    --data-dir /path/to/shhs \
    --checkpoint ./sleep_apnea/outputs/vit_bilstm_shhs/checkpoint_best.pt \
    --eval-only
```

## Expected Output Structure

```
sleep_apnea/
├── outputs/
│   └── cnn_baseline_shhs/
│       ├── checkpoint_best.pt
│       ├── checkpoint_last.pt
│       ├── results.json
│       └── logs/
├── checkpoints/
│   └── cnn_baseline_ssl_pretrain/
│       ├── pretrain_epoch_10.pt
│       └── pretrain_final.pt
└── logs/
```

## Training Timeline (RTX 5050 6GB)

| Model | SSL | Epochs | Batch Size | Expected Time |
|-------|-----|--------|------------|---------------|
| CNN Baseline | No | 30 | 64 | ~30 min |
| CNN Baseline | Yes | 100+30 | 64 | ~2 hours |
| ResNet18 TL | No | 30 | 64 | ~1 hour |
| ResNet18 TL | Yes | 100+30 | 64 | ~3 hours |
| ViT+BiLSTM | No | 30 | 32 | ~3 hours |
| ViT+BiLSTM | Yes | 100+30 | 32 | ~8 hours |

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python -m sleep_apnea.run_apnea --batch-size 16 ...

# Or use gradient accumulation (modify config)
```

### Dataset Not Found

```bash
# For SHHS, ensure NSRR approval is complete
# For Apnea-ECG, verify file structure:
ls /path/to/apnea-ecg/*.dat  # Should show a01.dat, a02.dat, etc.
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Verify timm installation (required for ViT/ResNet)
python -c "import timm; print(timm.__version__)"
```

## Next Steps

1. **Start with CNN baseline** to verify pipeline works
2. **Apply for SHHS access** immediately (takes 1-2 weeks)
3. **Use Apnea-ECG** for development while waiting
4. **Run full experiments** once SHHS access is granted

## See Also

- Full implementation plan: `../../SLEEP_APNEA_CLASSIFICATION_PLAN.md`
- Model documentation: `./models/`
- Configuration options: `./configs/default_config.yaml`
