# Hybrid Fusion Models - Multi-Modal Sleep Staging

## Overview

This document describes the three fusion architectures that combine **1D raw EEG signals** with **2D scalogram representations** for improved sleep stage classification.

**Date:** March 19, 2026  
**Status:** Phase 2 Complete ✅

---

## 🎯 Research Motivation

### Why Multi-Modal Fusion?

1. **Complementary Information:**
   - **1D Raw EEG:** Temporal dynamics, spike timing, phase information
   - **2D Scalograms:** Frequency patterns, time-frequency relationships, spectral signatures

2. **Biological Plausibility:**
   - Human brain processes both temporal and spectral information simultaneously
   - Sleep experts look at both time-domain waveforms AND frequency patterns

3. **Robustness:**
   - If one modality fails (e.g., noisy raw signal), the other can compensate
   - Ensemble effect reduces variance and improves generalization

---

## 📊 Fusion Architectures

### **1. Early Fusion (Feature Concatenation)**

```
┌─────────────────────────────────────────────────────────────┐
│                    EARLY FUSION                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1D Raw EEG (6, 3000)                                        │
│       ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ SNN-1D Encoder                                       │    │
│  │ (Temporal Pattern Extraction)                        │    │
│  └─────────────────────────────────────────────────────┘    │
│       ↓                                                      │
│  Features (128-d) ────────────┐                              │
│                                │                             │
│                                ├→ Concatenate → FC → Output  │
│                                │                             │
│  Features (512-d) ────────────┘                              │
│       ↑                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ SNN-2D / Quantum-2D Encoder                          │    │
│  │ (Spectral Pattern Extraction)                        │    │
│  └─────────────────────────────────────────────────────┘    │
│       ↑                                                      │
│  2D Scalogram (3, 224, 224)                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Advantages:**
- ✅ Simple and effective
- ✅ Learns cross-modal correlations end-to-end
- ✅ Single model training

**Disadvantages:**
- ❌ Requires both modalities at inference
- ❌ Cannot use pre-trained models directly
- ❌ Higher memory (need to store both feature sets)

**Best For:** Maximum accuracy when computational resources are available

---

### **2. Late Fusion (Ensemble Averaging)**

```
┌─────────────────────────────────────────────────────────────┐
│                    LATE FUSION                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1D Raw EEG (6, 3000)        2D Scalogram (3, 224, 224)     │
│       ↓                            ↓                         │
│  ┌──────────────────┐      ┌──────────────────┐             │
│  │ SNN-1D           │      │ SNN-2D / Quantum │             │
│  │ Classifier       │      │ Classifier       │             │
│  └──────────────────┘      └──────────────────┘             │
│       ↓                        ↓                             │
│  Logits (5-d)              Logits (5-d)                      │
│       ↓                        ↓                             │
│  Softmax                   Softmax                           │
│       ↓                        ↓                             │
│  Probs (5-d) ──→ Weighted Average → Final Probs             │
│       ↑              ↑                                       │
│       └──── w1 ──────┘                                       │
│       └──── w2 ─────────────────┘                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Advantages:**
- ✅ Can use pre-trained 1D and 2D models
- ✅ Interpretable branch weights (w1, w2)
- ✅ Robust to modality failure
- ✅ Can run single modality if needed

**Disadvantages:**
- ❌ No cross-modal feature learning
- ❌ Requires training two separate models
- ❌ Slower inference (two forward passes)

**Best For:** Interpretability and using existing pre-trained models

---

### **3. Gated Fusion (Confidence-Based Routing) ⭐ NOVEL**

```
┌─────────────────────────────────────────────────────────────┐
│                    GATED FUSION                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1D Raw EEG (6, 3000)                                        │
│       ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ SNN-1D Encoder + Confidence Estimator                 │    │
│  └─────────────────────────────────────────────────────┘    │
│       ↓                                                      │
│  Features (128-d) + Confidence Score (0-1)                  │
│       ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              GATING MECHANISM                        │    │
│  │                                                      │    │
│  │  if confidence > 0.7:                                │    │
│  │    → Use 1D only (FAST PATH) ⚡                      │    │
│  │  else:                                               │    │
│  │    → Activate 2D branch (ACCURATE PATH) 🎯          │    │
│  │    → Fuse features adaptively                        │    │
│  └─────────────────────────────────────────────────────┘    │
│       ↓                                                      │
│       ├─────────────────┐                                    │
│       │                 │                                    │
│  [High Conf]       [Low Conf]                                │
│       │                 │                                    │
│  1D Classifier     ┌────┴─────┐                             │
│       ↓            │           │                             │
│  Output        2D Branch    Fuse                            │
│                (Activated)   ↓                               │
│                           Output                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Advantages:**
- ✅ **Dynamic computation:** Easy cases use fast 1D path
- ✅ **Accuracy:** Hard cases get full 1D+2D processing
- ✅ **Efficiency:** Average inference time reduced
- ✅ **Novel contribution:** Paper-worthy innovation

**Disadvantages:**
- ❌ More complex training
- ❌ Need to tune confidence threshold
- ❌ Two-stage training may be needed

**Best For:** Real-time applications with variable computational budget

---

## 🔧 Implementation Details

### **Early Fusion**

```python
from src.models.fusion import create_early_fusion

model = create_early_fusion(
    num_classes=5,
    dim_1d=128,      # From SNN-1D final features
    dim_2d=512,      # From SNN-2D avgpool
    fusion_dim=256,  # Hidden dimension after fusion
)

# Training requires both modalities
logits = model(features_1d, features_2d)
```

**Architecture:**
- Input: `(128-d, 512-d)` concatenated → `(640-d)`
- FC1: `640 → 256` + BN + ReLU + Dropout(0.3)
- FC2: `256 → 128` + BN + ReLU + Dropout(0.15)
- Output: `128 → 5` (sleep stages)

---

### **Late Fusion**

```python
from src.models.fusion import create_late_fusion

# Option A: Learnable weights (recommended)
model = create_late_fusion(num_classes=5)

# Option B: Fixed weights
model = create_late_fusion(num_classes=5, weight_1d=0.6, weight_2d=0.4)

# Training
logits = model(logits_1d, logits_2d)

# Get learned weights
w1, w2 = model.get_weights()  # e.g., (0.65, 0.35)
```

**Weight Interpretation:**
- `w1 > w2`: 1D raw signal more informative
- `w2 > w1`: 2D scalogram more informative
- `w1 ≈ w2`: Both modalities equally important

---

### **Gated Fusion**

```python
from src.models.fusion import create_gated_fusion

model = create_gated_fusion(
    num_classes=5,
    dim_1d=128,
    dim_2d=512,
    confidence_threshold=0.7,  # Tune this
    gate_type='adaptive',      # 'hard', 'soft', or 'adaptive'
)

# Training
logits, gate_info = model(features_1d, features_2d)

# Inspect gating decisions
confidence = gate_info['confidence']  # (B, 1)
use_2d = gate_info.get('use_2d')      # For hard gating
```

**Gate Types:**

| Type | Behavior | Formula |
|------|----------|---------|
| `hard` | Binary routing | `output = gate ? 2D_fusion : 1D_only` |
| `soft` | Continuous blending | `output = conf·1D + (1-conf)·2D` |
| `adaptive` | Learned non-linear | `output = conf²·1D + (1-conf²)·2D` |

---

## 📈 Expected Performance

### **Hypotheses:**

| Model | Accuracy | F1-macro | Inference Time | Efficiency |
|-------|----------|----------|----------------|------------|
| **SNN-1D only** | 82-85% | 60-70% | **Fast** (1x) | **100%** |
| **SNN-2D only** | 85-88% | 65-72% | Slow (4x) | 25% |
| **Early Fusion** | **88-90%** | **70-75%** | Slow (4x) | 25% |
| **Late Fusion** | 87-89% | 68-73% | Slow (4x) | 25% |
| **Gated Fusion** | **88-90%** | **70-75%** | **Medium (2x)** | **50%** ⭐ |

**Key Insight:** Gated fusion achieves same accuracy as early fusion but with **2x speedup** by routing easy cases through 1D-only path!

---

## 🎯 Research Contributions

### **Paper-Worthy Insights:**

1. **Multi-modal sleep staging:** First work to systematically compare 1D vs 2D vs fusion for SNNs
2. **Gated fusion for SNNs:** Novel adaptive routing mechanism for efficient inference
3. **Modality analysis:** Quantify when 1D suffices vs when 2D is needed
4. **Efficiency-accuracy tradeoff:** Gated fusion achieves Pareto optimality

### **Experiments to Run:**

1. **Ablation Study:**
   - 1D only, 2D only, Early Fusion, Late Fusion, Gated Fusion
   - Compare accuracy, F1, inference time, memory

2. **Confidence Analysis:**
   - What percentage of samples use 1D-only path in gated fusion?
   - Are high-confidence samples actually easier?
   - Which sleep stages benefit most from 2D?

3. **Cross-Modality Comparison:**
   - SNN-1D + SNN-2D fusion
   - SNN-1D + Quantum-2D fusion
   - Transformer-1D + Transformer-2D fusion

4. **Failure Mode Analysis:**
   - When does 1D fail but 2D succeeds?
   - Visualize scalograms for low-confidence samples
   - Identify spectral patterns missed by 1D

---

## 🚀 How to Train

### **In pipeline.py:**

```bash
# Early Fusion
python pipeline.py --experiment snn_fusion_early

# Late Fusion
python pipeline.py --experiment snn_fusion_late

# Gated Fusion (recommended!)
python pipeline.py --experiment snn_fusion_gated
```

### **Training Configuration:**

All fusion models automatically use:
- **Data mode:** `both` (loads both 1D and 2D data)
- **Batch size:** 64 (reduced from 128 due to memory)
- **Learning rate:** 3e-4 (SNN-optimized)
- **Epochs:** 50
- **Mixed precision:** False (FP32 for stability)

---

## 📊 Monitoring

### **During Training:**

```python
# For gated fusion
for batch in dataloader:
    x_1d, x_2d, labels = batch
    logits, gate_info = model(features_1d, features_2d)
    
    # Log gating statistics
    avg_conf = gate_info['confidence'].mean().item()
    pct_2d = (gate_info['confidence'] < 0.7).float().mean().item()
    
    wandb.log({
        'avg_confidence': avg_conf,
        'pct_using_2d': pct_2d,
    })
```

**Expected gating behavior:**
- 60-70% samples: High confidence → 1D only (fast)
- 30-40% samples: Low confidence → 1D+2D fusion (accurate)

---

## 📝 Summary

| Aspect | Status |
|--------|--------|
| **Early Fusion** | ✅ Implemented |
| **Late Fusion** | ✅ Implemented |
| **Gated Fusion** | ✅ Implemented |
| **Pipeline Integration** | ✅ Complete |
| **Documentation** | ✅ Complete |
| **Testing** | ⏳ Pending |

---

**Next Step:** Test fusion models and compare with single-modality baselines!

**Expected Outcome:** Gated fusion achieves 88-90% accuracy with 2x speedup over full 1D+2D ensemble.
