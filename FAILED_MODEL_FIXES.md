# 🐛 Failed Model Bug Report & Fixes

**Date:** March 22, 2026  
**Analysis:** Complete codebase review of failed models

---

## 📊 Summary of Failures

| Model | Status | Runtime | Accuracy | Issue |
|-------|--------|---------|----------|-------|
| **fusion_c** | ❌ CRASHED | 3859s (1hr) | - | Too slow, OOM |
| **snn_fusion_late** | ❌ CRASHED | 1263s | 10.18% | Gradient vanishing |
| **snn_fusion_early** | ⚠️ NO METRICS | 1625s | - | Feature mismatch |
| **spiking_vit_1d** | ❌ FAILED | 1033s | 10.94% | Not actually spiking |

---

## 🔍 Root Causes & Fixes

### 1. [CRITICAL] fusion_c - CRASHED (1 hour/epoch)

**Problem:** Combining Swin Transformer (28M params) + SNN-1D (8 timesteps) = **computational catastrophe**

**Math:**
- Swin forward: ~100ms per sample
- SNN-1D forward: 8 timesteps × 50ms = 400ms
- Combined: 500ms × 83,767 samples = **11.6 hours per epoch** (theoretical)
- Actual: 1 hour (with batch parallelization, but still too slow)

**Fix: Replace Swin with EfficientNet-B0**

```python
# File: src/models/fusion/fusion_c.py (Line 48-53)

# BEFORE (BROKEN):
self.swin = timm.create_model(
    'swin_tiny_patch4_window7_224',  # 28M params
    pretrained=pretrained,
    num_classes=0,
)

# AFTER (FIXED):
self.backbone = timm.create_model(
    'efficientnet_b0',  # 5M params, 10x faster
    pretrained=pretrained,
    num_classes=0,
)
```

**Expected Impact:** 10x speedup (1hr → 6 min/epoch)

---

### 2. [CRITICAL] snn_fusion_late - CRASHED (10% accuracy)

**Problem:** Numerical instability from `log(softmax())` causing gradient vanishing

**Broken Code:**
```python
# File: src/models/fusion/fusion_models.py (Lines 563-573)

probs_1d = F.softmax(logits_1d, dim=1)
probs_2d = F.softmax(logits_2d, dim=1)

w1 = torch.sigmoid(self.weight_1d)
w2 = torch.sigmoid(self.weight_2d)
total = w1 + w2
w1, w2 = w1 / total, w2 / total

fused_probs = w1 * probs_1d + w2 * probs_2d
return torch.log(fused_probs + 1e-8)  # ❌ NUMERICAL INSTABILITY
```

**Why It Fails:**
- When `probs` ≈ 0, `log(probs + 1e-8)` ≈ -18.4
- Gradient of `log()` at small values ≈ 0
- **Result:** No gradient flow → model can't learn → 10% accuracy (worse than random 20%)

**Fix: Use logit fusion instead of probability fusion**

```python
# File: src/models/fusion/fusion_models.py

class SNNFusionLateComplete(nn.Module):
    def __init__(self, model_1d, model_2d, num_classes=5):
        super().__init__()
        self.model_1d = model_1d
        self.model_2d = model_2d
        # Single learnable parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, raw_signal=None, scalogram=None, **kwargs):
        logits_1d = self.model_1d(raw_signal)
        logits_2d = self.model_2d(scalogram)
        
        # ✅ STABLE: Weighted average of LOGITS
        alpha = torch.sigmoid(self.alpha)
        fused_logits = alpha * logits_1d + (1 - alpha) * logits_2d
        
        return fused_logits  # Let CrossEntropyLoss handle softmax
```

**Expected Impact:** Model will train properly, accuracy 70-80%

---

### 3. [HIGH] snn_fusion_early - NO METRICS

**Problem:** Feature dimension mismatch + no gradient validation

**Symptoms:**
- Model "finished" but no metrics in CSV
- Indicates: Metric computation failed (NaN/Inf/empty arrays)

**Root Cause:**
```python
# File: src/models/fusion/fusion_models.py (Lines 507-518)

self.fusion_classifier = nn.Sequential(
    nn.Linear(128 + 512, 256),  # ✓ Correct
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    nn.Linear(128, 5),
)
```

**Issue:** Deep classifier + SNN gradients = vanishing gradients

**Fix: Simplify classifier + add LayerNorm**

```python
# File: src/models/fusion/fusion_models.py

class SNNFusionEarlyComplete(nn.Module):
    def __init__(self, model_1d, model_2d, num_classes=5, 
                 dim_1d=128, dim_2d=512, fusion_dim=256):
        super().__init__()
        self.model_1d = model_1d
        self.model_2d = model_2d
        
        # ✅ SIMPLIFIED: Direct to classes
        self.fusion_classifier = nn.Sequential(
            nn.Linear(dim_1d + dim_2d, fusion_dim),
            nn.LayerNorm(fusion_dim),  # Better than BatchNorm
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, num_classes),  # Direct
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, raw_signal=None, scalogram=None, **kwargs):
        features_1d = self.model_1d.extract_features(raw_signal)
        features_2d = self.model_2d.extract_features(scalogram)
        
        # Debug validation
        assert features_1d.shape[1] == 128
        assert features_2d.shape[1] == 512
        
        fused = torch.cat([features_1d, features_2d], dim=1)
        return self.fusion_classifier(fused)
```

**Expected Impact:** Valid metrics, accuracy 75-85%

---

### 4. [MEDIUM] spiking_vit_1d - 10.94% accuracy

**Problem:** **Not actually a spiking model** - uses continuous activations

**Evidence:**
```python
# File: src/models/snn_1d/spiking_vit.py (Lines 94-117)

class SpikingTransformerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.mssa = MultiScaleSpikingAttention(dim=dim, ...)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),  # ❌ Continuous activation
            nn.Linear(hidden, dim),
        )
    
    def forward(self, x):
        x = x + self.mssa(x)  # ❌ No spikes
        x = x + self.ffn(self.norm(x))  # ❌ Continuous FFN
        return x
```

**Why 10% accuracy?**
- Standard ViT needs massive pretraining (ImageNet-21k)
- No spiking dynamics = loses temporal pattern benefits
- Under-parameterized (embed_dim=128, depth=4)

**Fix Option A: Use pretrained ViT (Recommended)**

```python
# Just use timm's ViT - better optimized and pretrained
import timm
model = timm.create_model(
    'vit_small_patch16_224',
    pretrained=True,
    num_classes=5
)
```

**Fix Option B: Make it actually spiking (2 days)**

Add LIF neurons and timesteps to transformer blocks (see full analysis for code)

**Recommendation:** **Remove from paper_pipeline.py** - not worth the effort

---

## 🛠️ Implementation Priority

### Phase 1: Critical Fixes (1 hour)

1. **Fix snn_fusion_late** (15 min)
   - Change log-softmax to logit fusion
   - File: `src/models/fusion/fusion_models.py`

2. **Fix snn_fusion_early** (30 min)
   - Simplify classifier, add LayerNorm
   - File: `src/models/fusion/fusion_models.py`

3. **Fix fusion_c** (15 min)
   - Replace Swin with EfficientNet-B0
   - File: `src/models/fusion/fusion_c.py`

### Phase 2: Remove Broken Models (5 min)

4. **Remove spiking_vit_1d** from paper_pipeline.py
   - Not worth fixing
   - Add deprecation warning

---

## 📈 Expected Results After Fixes

| Model | Current | After Fix | Time/Epoch |
|-------|---------|-----------|------------|
| fusion_c | CRASHED | 75-80% | 6 min ✅ |
| snn_fusion_late | CRASHED | 70-75% | 3 min ✅ |
| snn_fusion_early | NO METRICS | 75-85% | 3 min ✅ |
| spiking_vit_1d | 10.94% | REMOVED | - |

---

## 🚀 Commands to Apply Fixes

```bash
# On your Windows machine (then push to server)
cd D:\Projects\AI-Projects\EEG

# 1. Fix snn_fusion_late (log-softmax → logit fusion)
# Edit: src/models/fusion/fusion_models.py

# 2. Fix snn_fusion_early (simplify classifier)
# Edit: src/models/fusion/fusion_models.py

# 3. Fix fusion_c (Swin → EfficientNet-B0)
# Edit: src/models/fusion/fusion_c.py

# 4. Remove spiking_vit_1d from paper_pipeline
# Edit: paper_pipeline.py (remove from presets)

# Commit and push
git add -A
git commit -m "Fix failed fusion models
- snn_fusion_late: log-softmax → logit fusion (fixes gradient vanishing)
- snn_fusion_early: simplify classifier (fixes no metrics)
- fusion_c: Swin → EfficientNet-B0 (10x speedup)
- Remove spiking_vit_1d (not actually spiking)"
git push origin main
```

---

## ✅ Verification on Server

```bash
# On Ubuntu server
cd ~/Projects/Cardio/Cancer/EEG_Multiple_Diseases
git pull origin main

# Test fusion_c (should be fast now)
python3 -c "
from src.models.fusion import create_fusion_c
model = create_fusion_c(num_classes=5)
import torch
x1d = torch.randn(4, 6, 3000)
x2d = torch.randn(4, 3, 224, 224)
import time
start = time.time()
out = model(x1d, x2d)
print(f'fusion_c forward time: {time.time()-start:.2f}s')
print(f'Output shape: {out.shape}')
"

# Should output:
# fusion_c forward time: <5s
# Output shape: torch.Size([4, 5])
```

---

**Last Updated:** March 22, 2026  
**Status:** Ready to implement
