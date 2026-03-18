# Server Training Guide - A100 GPU

## 🚀 Quick Start (Copy-Paste Commands)

### **Option 1: Full Training Pipeline (All Experiments)**

```bash
# 1. Navigate to project
cd /path/to/EEG/project

# 2. Update paths in script
nano train_snn_fusion.sh  # Edit PROJECT_DIR and DATA_DIR

# 3. Make executable
chmod +x train_snn_fusion.sh

# 4. Run with nohup (train and forget!)
nohup ./train_snn_fusion.sh > logs/training_master.log 2>&1 &

# 5. Check it's running
ps aux | grep train_snn_fusion
```

---

### **Option 2: Just Gated Fusion (Recommended - Main Contribution)**

```bash
# 1. Navigate to project
cd /path/to/EEG/project

# 2. Update paths in script
nano quick_train_gated_fusion.sh  # Edit DATA_DIR

# 3. Make executable
chmod +x quick_train_gated_fusion.sh

# 4. Run with nohup
nohup ./quick_train_gated_fusion.sh > logs/gated_fusion.log 2>&1 &

# 5. Check it's running
ps aux | grep gated_fusion
```

---

### **Option 3: Direct Command (No Script)**

```bash
# Single command - copy, paste, run!
cd /path/to/EEG/project && \
export CUDA_VISIBLE_DEVICES=0 && \
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 && \
nohup python pipeline.py \
    --experiment snn_fusion_gated \
    --data-dir /path/to/datasets/BOAS \
    --output-dir ./outputs \
    --epochs 50 \
    --batch-size 128 \
    --learning-rate 3e-4 \
    --num-workers 8 \
    > logs/gated_fusion_nohup.log 2>&1 &
```

---

## 📊 Monitoring Commands

### **Check if Training is Running**

```bash
# See all Python processes
ps aux | grep python

# See specific training
ps aux | grep pipeline.py

# Check GPU usage
watch -n 1 nvidia-smi

# Check GPU usage (simpler)
gpustat -i 1
```

---

### **View Training Logs**

```bash
# === MOST IMPORTANT ===
# Follow gated fusion training in real-time
tail -f logs/gated_fusion.log

# Follow all logs
tail -f logs/training_master.log

# Last 50 lines
tail -n 50 logs/gated_fusion.log

# With grep for specific info
tail -f logs/gated_fusion.log | grep -E "(Epoch|Accuracy|F1|loss)"
```

---

### **Advanced Monitoring**

```bash
# See training progress (grep for epoch info)
tail -f logs/gated_fusion.log | grep "Epoch"

# See validation results
tail -f logs/gated_fusion.log | grep -E "(val_acc|val_f1|best_epoch)"

# See errors
tail -f logs/gated_fusion.log | grep -i error

# See warnings
tail -f logs/gated_fusion.log | grep -i warn

# Count completed epochs
grep "Epoch.*completed" logs/gated_fusion.log | wc -l
```

---

### **Check Results**

```bash
# View latest results
cat outputs/results/snn_fusion_gated_results.json | python -m json.tool

# Or use the analysis script
python analyze_results.py

# List all result files
ls -lh outputs/results/
```

---

## 🔍 Useful One-Liners

```bash
# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Check training time
grep "duration_seconds" outputs/results/snn_fusion_gated_results.json

# Extract accuracy from results
python -c "import json; d=json.load(open('outputs/results/snn_fusion_gated_results.json')); print(f\"Accuracy: {d['metrics']['best_accuracy']:.4f}\")"

# See all running experiments
ps aux | grep "pipeline.py" | grep -v grep

# Kill a specific training
pkill -f "pipeline.py.*snn_fusion_gated"

# Check disk usage
du -sh outputs/ logs/

# Check if dataset is downloaded
ls -lh /path/to/datasets/BOAS/
```

---

## 📈 Expected Training Output

### **What to Look For:**

```
[SNN-1D MODE] Using 1D SNN-optimized hyperparameters:
  - Model type: snn_1d (raw EEG signal processing)
  - Learning rate: 0.0003 (reduced for BPTT stability)
  - Warmup epochs: 10 (longer for spike dynamics)
  - Timesteps: 25 (increased from 4/8)
  ...

Epoch 1/50: 100%|██████████| 1234/1234 [05:23<00:00, 3.82it/s, loss=1.23, acc=0.45]
Epoch 1/50 - val_loss: 1.15, val_acc: 0.52, val_f1: 0.48
Epoch 2/50: 100%|██████████| 1234/1234 [05:21<00:00, 3.84it/s, loss=1.08, acc=0.58]
...
Epoch 10/50: Best model saved (val_acc=0.78, val_f1=0.72)
...
Training completed! Best accuracy: 0.85, Best F1: 0.78
```

### **Gated Fusion Specific:**

```
[GATED FUSION] Confidence-based routing active
  - Threshold: 0.7
  - Gate type: adaptive
Average confidence: 0.73
Using 2D branch: 35.2%  ← Should be 30-40% (60-70% use 1D-only!)
```

---

## ⚠️ Troubleshooting

### **Training Crashes Immediately**

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check dataset path
ls -lh /path/to/datasets/BOAS/

# Check memory
nvidia-smi  # Should show >40GB free on A100
```

### **Out of Memory**

```bash
# Reduce batch size
--batch-size 64  # Instead of 128

# Reduce workers
--num-workers 4  # Instead of 8
```

### **Training Too Slow**

```bash
# Check GPU utilization
watch -n 1 nvidia-smi  # Should be >80%

# Increase workers if CPU bottleneck
--num-workers 12

# Check if using A100
nvidia-smi --query-gpu=name --format=csv,noheader
```

### **NaN Loss**

```bash
# Check learning rate (should be 3e-4 for SNNs)
--learning-rate 3e-4

# Check mixed precision (should be disabled for SNNs)
# This is automatic in pipeline.py for SNN experiments
```

---

## 📊 Complete Training Session Example

```bash
# === ON SERVER ===

# 1. Connect
ssh your-server

# 2. Navigate
cd /path/to/EEG/project

# 3. Create screen session (optional but recommended)
screen -S snn_training

# 4. Run training
nohup ./quick_train_gated_fusion.sh > logs/gated_fusion.log 2>&1 &

# 5. Detach from screen (Ctrl+A, then D)

# === CHECK LATER ===

# 6. Reconnect
ssh your-server
cd /path/to/EEG/project

# 7. Reattach screen
screen -r snn_training

# 8. Check progress
tail -f logs/gated_fusion.log

# 9. Check GPU
watch -n 5 nvidia-smi

# 10. Check results
python analyze_results.py
```

---

## 🎯 Quick Reference Card

```bash
# START TRAINING
nohup python pipeline.py --experiment snn_fusion_gated > logs/gated_fusion.log 2>&1 &

# MONITOR
tail -f logs/gated_fusion.log

# CHECK GPU
watch -n 1 nvidia-smi

# SEE PROGRESS
grep "Epoch" logs/gated_fusion.log | tail -20

# CHECK RESULTS
cat outputs/results/snn_fusion_gated_results.json | python -m json.tool

# KILL IF NEEDED
pkill -f "pipeline.py.*snn_fusion_gated"
```

---

## 📈 Expected Timeline (A100)

| Experiment | Epochs | Time | Accuracy |
|------------|--------|------|----------|
| snn_1d_lif | 50 | ~2 hours | 82-85% |
| snn (2D) | 50 | ~6 hours | 85-88% |
| snn_fusion_early | 50 | ~8 hours | 88-90% |
| snn_fusion_late | 50 | ~8 hours | 87-89% |
| **snn_fusion_gated** | **50** | **~5 hours** | **88-90%** |

**Total for all experiments:** ~29 hours (run overnight!)

---

**Pro Tip:** Start training before you go to sleep, check results in the morning! 🌙→☀️
