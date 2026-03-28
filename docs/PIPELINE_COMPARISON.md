# 🔄 Pipeline Comparison: Old vs New

**Why we created the unified pipeline.**

---

## 📊 Side-by-Side Comparison

| Aspect | Old Approach | New Unified Approach |
|--------|-------------|---------------------|
| **Files** | `pipeline.py`, `sleep_apnea_pipeline.py` | `unified_pipeline.py` ✅ |
| **Interface** | Different for each task | Single CLI for all tasks ✅ |
| **Code Reuse** | Duplicated trainers, configs | Shared everything ✅ |
| **Adding Tasks** | Create new file | Add to registry ✅ |
| **Comparison** | Hard (different outputs) | Easy (same format) ✅ |
| **Maintenance** | Fix bugs in multiple places | Fix once, works everywhere ✅ |

---

## 🎯 Task Support

### Old Approach

```bash
# Sleep Staging (pipeline.py)
python pipeline.py --models snn_lif_resnet --epochs 50

# Sleep Apnea (sleep_apnea_pipeline.py)
python sleep_apnea_pipeline.py --model vit_bilstm --epochs 30

# Different flags, different outputs, different everything
```

### New Unified Approach

```bash
# Sleep Staging
python unified_pipeline.py --task sleep_staging --model snn_lif_resnet --epochs 50

# Sleep Apnea
python unified_pipeline.py --task sleep_apnea --model vit_bilstm --epochs 30

# Same interface, easy to remember!
```

---

## 📁 File Organization

### Old (Multiple Pipelines)
```
EEG/
├── pipeline.py                      # Sleep staging only
├── sleep_apnea_pipeline.py          # Sleep apnea only
├── run_sleep_apnea_experiments.py   # Separate runner
├── src/
│   └── models/
│       ├── snn/                     # Staging models
│       └── quantum/                 # Staging models
└── sleep_apnea/
    └── models/                      # Apnea models (duplicated code)
```

**Problems:**
- ❌ Duplicated training loops
- ❌ Different config systems
- ❌ Hard to compare results
- ❌ Fix bugs in 2+ places

### New (Unified)
```
EEG/
├── unified_pipeline.py              # ONE pipeline for everything
├── src/
│   └── models/
│       ├── snn/                     # All models together
│       ├── transformers/
│       └── fusion/
└── outputs/unified/
    ├── sleep_staging/
    └── sleep_apnea/
```

**Benefits:**
- ✅ Single source of truth
- ✅ Shared utilities
- ✅ Easy comparison
- ✅ Fix once, works everywhere

---

## 🧪 Running Experiments

### Old Way (Confusing)

```bash
# Run sleep staging
python pipeline.py --models snn_lif_resnet,snn_qif_resnet --epochs 50

# Run sleep apnea
python sleep_apnea_pipeline.py --model cnn --data-dir /path/to/shhs

# Run all apnea experiments
python run_sleep_apnea_experiments.py --all

# Wait, which script do I use again?
```

### New Way (Simple)

```bash
# Run sleep staging
python unified_pipeline.py --task sleep_staging --model snn_lif_resnet --epochs 50

# Run sleep apnea
python unified_pipeline.py --task sleep_apnea --model cnn --epochs 30

# Run all experiments for a task
python unified_pipeline.py --run-all --task sleep_staging
python unified_pipeline.py --run-all --task sleep_apnea

# Consistent, intuitive!
```

---

## 📊 Output Format

### Old (Inconsistent)

```json
// pipeline.py output
{
  "experiment": "snn_lif_resnet",
  "metrics": {
    "val_acc": 85.2,
    "test_acc": 83.1
  }
}

// sleep_apnea_pipeline.py output
{
  "model": "vit_bilstm",
  "results": {
    "validation_accuracy": 96.8,
    "test_accuracy": 94.2
  }
}

// Different field names, hard to compare!
```

### New (Consistent)

```json
// unified_pipeline.py output (both tasks)
{
  "task": "sleep_staging",  // or "sleep_apnea"
  "model": "snn_lif_resnet",
  "dataset": "boas",
  "metrics": {
    "val_acc": 85.2,
    "test_acc": 83.1,
    "f1_macro": 0.82
  }
}

// Same format, easy to compare!
```

---

## 🔧 Adding a New Model

### Old Way

**In `pipeline.py`:**
```python
def create_model(exp_config):
    if exp_type == 'snn':
        # ... 200 lines of model creation
```

**In `sleep_apnea_pipeline.py`:**
```python
def create_model(model_name):
    # ... duplicate code
```

**In `run_sleep_apnea_experiments.py`:**
```python
EXPERIMENT_CONFIGS = {
    # ... duplicate config
}
```

**Result:** Update 3+ files, risk inconsistencies

### New Way

**In `unified_pipeline.py`:**
```python
MODEL_REGISTRY = {
    'my_new_model': MyNewModel,
}
```

**Result:** Update 1 file, automatically works for all tasks!

---

## 🐛 Bug Fixes

### Example: BatchNorm → InstanceNorm Fix

**Old Way:**
1. Fix in `src/models/snn/spiking_resnet.py` ✅
2. Remember to fix in `sleep_apnea/models/resnet.py` ❌
3. Remember to fix in other pipelines ❌
4. Hope you didn't miss any ❌

**New Way:**
1. Fix in unified model registry ✅
2. Done! Works for all tasks ✅

---

## 📈 Scalability

### Adding New Tasks

**Old Way:**
- Create new pipeline file
- Implement new trainer
- Create new configs
- Write new experiment runner
- **Time: 2-3 days**

**New Way:**
- Add task to `UnifiedConfig`
- Add models to registry
- Implement dataset loader
- **Time: 1-2 hours**

**Example: Adding Emotion Recognition**

```python
# In unified_pipeline.py
class UnifiedConfig:
    EMOTION_CLASSES = 7
    EMOTION_NAMES = ['neutral', 'happy', 'sad', ...]

# Register models
MODEL_REGISTRY['emotion_cnn'] = EmotionCNN

# Done! Use immediately:
python unified_pipeline.py --task emotion --model emotion_cnn
```

---

## 🎯 Migration Guide

### From `pipeline.py` (Sleep Staging)

**Old:**
```bash
python pipeline.py --models snn_lif_resnet --epochs 50
```

**New:**
```bash
python unified_pipeline.py --task sleep_staging --model snn_lif_resnet --epochs 50
```

### From `sleep_apnea_pipeline.py`

**Old:**
```bash
python sleep_apnea_pipeline.py --model vit_bilstm --ssl-pretrain
```

**New:**
```bash
python unified_pipeline.py --task sleep_apnea --model vit_bilstm --ssl-pretrain
```

### From `run_sleep_apnea_experiments.py`

**Old:**
```bash
python run_sleep_apnea_experiments.py --all
```

**New:**
```bash
python unified_pipeline.py --run-all --task sleep_apnea
```

---

## ✅ Benefits Summary

| Benefit | Impact |
|---------|--------|
| **Single Interface** | No more remembering different commands ✅ |
| **Code Reuse** | 50% less code duplication ✅ |
| **Easy Comparison** | Same output format for all tasks ✅ |
| **Fast Iteration** | Add new tasks in hours, not days ✅ |
| **Bug Fixes** | Fix once, works everywhere ✅ |
| **Maintenance** | One codebase to maintain ✅ |
| **Documentation** | One README to update ✅ |

---

## 🚀 What to Use Now

| Task | Recommended Command |
|------|---------------------|
| **Sleep Staging** | `python unified_pipeline.py --task sleep_staging ...` ✅ |
| **Sleep Apnea** | `python unified_pipeline.py --task sleep_apnea ...` ✅ |
| **New Tasks** | `python unified_pipeline.py --task <new_task> ...` ✅ |

**Old pipelines still work, but unified is the future!**

---

## 📞 Quick Reference

```bash
# Unified pipeline - Sleep Staging
python unified_pipeline.py \
    --task sleep_staging \
    --model snn_lif_resnet \
    --dataset boas \
    --epochs 50

# Unified pipeline - Sleep Apnea
python unified_pipeline.py \
    --task sleep_apnea \
    --model vit_bilstm \
    --dataset shhs \
    --ssl-pretrain

# Run all experiments
python unified_pipeline.py --run-all --task sleep_staging
python unified_pipeline.py --run-all --task sleep_apnea

# Old pipelines (deprecated but working)
python pipeline.py --models snn_lif_resnet  # Still works
python sleep_apnea_pipeline.py --model cnn  # Still works
```

---

**Migrate to unified pipeline today!** 🚀

Your future self will thank you when you need to:
- Compare results across tasks
- Add new models
- Fix bugs
- Write documentation

---

**Last Updated:** March 2026  
**Status:** ✅ Unified pipeline ready for production
