# 🐛 Root Cause Analysis: SNN Validation Stuck at 13.6%

## Observations

1. **Training accuracy improves**: 27% → 40% → 46% (model IS learning)
2. **Validation accuracy stuck at exactly 13.6%** (not changing at all)
3. **13.6% ≈ class 0 (Wake) distribution** in BOAS dataset
4. **Same behavior across multiple runs** (deterministic, not random)

## Hypothesis

The model is **always predicting class 0** during validation because the output logits are all zeros, NaN, or negative infinity, and `outputs.max(1)` returns index 0 by default.

## Root Cause: BatchNorm Running Statistics

The most likely cause is that **BatchNorm running statistics are corrupted** during training:

### What's Happening:

1. **During training** (`model.train()`):
   - BatchNorm uses **batch statistics** (mean/var of current batch)
   - Running statistics are updated with momentum
   - Model works fine, accuracy improves

2. **During validation** (`model.eval()`):
   - BatchNorm switches to **running statistics**
   - If running_mean or running_var are NaN/Inf, output becomes NaN
   - Model outputs garbage, all predictions default to class 0

### Why BatchNorm Stats Get Corrupted:

The spike encoding produces **binary 0/1 values** (Poisson spikes):
```python
spike_mask = torch.rand_like(x_spike_prob) < x_spike_prob
x_t = x_spike_prob * spike_mask.float()  # Binary: 0 or x_spike_prob
```

When this goes through BatchNorm:
- Many zeros → running_mean becomes very small
- Binary values → running_var becomes very small
- Eventually running_var → 0 or negative (due to momentum updates)
- Division by ~0 in BatchNorm → NaN/Inf outputs

## Solution Options

### Option 1: Use BatchNorm in Training Mode Only (Quick Fix)

Force BatchNorm to use batch statistics even during eval:

```python
# In spiking_resnet.py forward method
def forward(self, x):
    # ... existing code ...
    
    # Force BatchNorm to use batch stats (not running stats)
    self.bn1.training = True
    for layer in self.layer1:
        if hasattr(layer, 'bn1'):
            layer.bn1.training = True
            layer.bn2.training = True
    
    # ... rest of forward ...
```

### Option 2: Remove BatchNorm from Spiking Layers (Better Fix)

BatchNorm doesn't work well with binary spike inputs. Replace with:
- GroupNorm (doesn't use running stats)
- LayerNorm (per-sample normalization)
- Or remove normalization entirely (spikes are already normalized)

```python
# Replace in SpikingConv2d and SpikingBasicBlock
self.bn = nn.GroupNorm(num_groups=8, num_channels=out_ch)
# or
self.bn = nn.Identity()  # Remove entirely
```

### Option 3: Fix Running Statistics Update (Proper Fix)

Clamp running_var to prevent it from going to zero:

```python
# Add hook to all BatchNorm layers
def clamp_running_var(module):
    if isinstance(module, nn.BatchNorm2d):
        module.register_buffer('running_var_clamped', torch.clamp(module.running_var, min=1e-4))
        # Use clamped version in forward
```

Or manually fix after each training step:

```python
# In training loop, after each batch
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.running_var = torch.clamp(module.running_var, min=1e-4)
```

### Option 4: Use InstanceNorm Instead (Recommended)

InstanceNorm normalizes per sample, not across batch:

```python
# Replace all BatchNorm2d with InstanceNorm2d
self.bn1 = nn.InstanceNorm2d(64, affine=True)
```

This works better for SNNs because:
- No running statistics to corrupt
- Works with any batch size
- More stable with binary inputs

## Recommended Fix

**Option 4 (InstanceNorm)** is the cleanest solution:

1. Replace `nn.BatchNorm2d` with `nn.InstanceNorm2d` in:
   - `SpikingResNet.__init__()` (bn1)
   - `SpikingConv2d.__init__()`
   - `SpikingBasicBlock.__init__()` (bn1, bn2)
   - Shortcut connections

2. Test - validation accuracy should now work correctly

## Files to Modify

- `src/models/snn/spiking_resnet.py` - Replace all BatchNorm2d
- `src/models/snn/spiking_vit.py` - Replace BatchNorm if present
- `src/models/snn_1d/snn_classifier.py` - Replace BatchNorm1d with InstanceNorm1d

## Verification

After fix, run a single epoch and check:
- Training accuracy should improve
- **Validation accuracy should also improve** (not stuck at 13.6%)
- Model outputs should be valid (not all zeros/NaN)
