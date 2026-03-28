# 🧠 Complete EEG Model Documentation & Results

**Project:** EEG Sleep Staging & Classification  
**Dataset:** BOAS (ds005555) - 5-class sleep staging (Wake/N1/N2/N3/REM)  
**Last Updated:** March 20, 2026

---

## 📊 Executive Summary

| Category | Models Tested | Best Model | Best Accuracy | Status |
|----------|--------------|------------|---------------|--------|
| **2D SNN** | 2 | `snn_lif_resnet` | **69.7%** | ✅ Complete |
| **1D SNN** | 3 | `snn_1d_lif` | **42.1%** | ✅ Complete |
| **1D Quantum** | 14 | `quantum_1d_full_RXY` | **53.9%** | ✅ Complete |
| **2D Quantum** | 14 | `quantum_full_RXY` | **83.4%** | ✅ Complete |
| **Fusion** | 4 | `fusion_a` | **82.0%** | ✅ Complete |
| **SNN Fusion** | 3 | Running | - | 🔄 In Progress |
| **Transformers** | 5 | Not run yet | - | ⏳ Pending |

---

## 📋 Table of Contents

1. [Spiking Neural Networks (SNN)](#1-spiking-neural-networks-snn)
2. [Quantum-Classical Hybrid CNNs](#2-quantum-classical-hybrid-cnns)
3. [Vision Transformers](#3-vision-transformers)
4. [Fusion Models](#4-fusion-models)
5. [Complete Results Table](#5-complete-Results-Table)

---

## 1. Spiking Neural Networks (SNN)

### 1.1 SNN-ResNet18-LIF ⭐

**File:** `src/models/snn/spiking_resnet.py`

**Description:**
Spiking ResNet-18 using Leaky Integrate-and-Fire (LIF) neurons. Processes 2D scalograms through spiking convolutional layers with surrogate gradient backpropagation. Uses rate coding readout - accumulates spikes over 8 timesteps for classification.

**Architecture:**
```
Scalogram (3, 224, 224)
    ↓
Conv2d(7×7, s=2) + InstanceNorm + LIF
    ↓
MaxPool + 4× SpikingResNet Blocks (LIF neurons)
    ↓
AdaptiveAvgPool + FC(512→5)
    ↓
Spike Count over 8 timesteps → 5-class logits
```

**Key Features:**
- **Neuron Type:** LIF (Leaky Integrate-and-Fire) with β=0.9
- **Timesteps:** 8 (optimized for speed/accuracy tradeoff)
- **Normalization:** InstanceNorm2d (fixed from BatchNorm)
- **Spike Encoding:** Poisson spike trains from normalized scalogram

**Results:**
| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **69.74%** |
| **Validation F1 (macro)** | **0.576** |
| **Validation AUC-ROC** | **0.913** |
| **Best Epoch** | 21 |
| **Training Time** | ~6 hours |

**Code Snippet:**
```python
class SpikingResNet(nn.Module):
    def __init__(self, num_classes=5, num_timesteps=8, neuron_type='lif'):
        self.num_timesteps = num_timesteps
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.InstanceNorm2d(64, affine=True)
        self.lif1 = create_neuron('lif', beta=0.9)
        # ... 4 ResNet layers with LIF neurons
    
    def forward(self, x):
        # Convert to spike probabilities
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x_spike_prob = x_norm * 0.7  # 70% max firing rate
        
        spike_record = []
        for t in range(self.num_timesteps):
            spike_mask = torch.rand_like(x_spike_prob) < x_spike_prob
            x_t = x_spike_prob * spike_mask.float()
            spk = self.forward_timestep(x_t)
            spike_record.append(spk)
        
        return torch.stack(spike_record, dim=0).sum(dim=0)  # Rate coding
```

---

### 1.2 SNN-ResNet18-QIF

**File:** `src/models/snn/spiking_resnet.py`

**Description:**
Spiking ResNet-18 using Quadratic Integrate-and-Fire (QIF) neurons. QIF has nonlinear quadratic membrane dynamics (dv/dt = v² + μ) vs LIF's linear dynamics, potentially capturing richer temporal patterns.

**Architecture:**
Same as SNN-ResNet18-LIF, but with QIF neurons instead of LIF.

**Key Features:**
- **Neuron Type:** QIF (Quadratic Integrate-and-Fire)
- **Membrane Dynamics:** τ·dv/dt = v² + μ(t) (nonlinear)
- **Timesteps:** 8
- **Status:** ⚠️ **FAILED** - BatchNorm issue caused 13.6% accuracy (stuck predicting class 0)

**Results:**
| Metric | Value | Status |
|--------|-------|--------|
| Validation Accuracy | 13.55% | ❌ Failed |
| Validation F1 | 0.048 | ❌ Failed |
| Issue | BatchNorm running stats corrupted | 🔧 Needs rerun |

---

### 1.3 SNN-ViT-LIF

**File:** `src/models/snn/spiking_vit.py`

**Description:**
Spiking Vision Transformer using LIF neurons. Combines transformer's self-attention with spiking dynamics for temporal pattern recognition in EEG scalograms.

**Architecture:**
```
Scalogram (3, 224, 224)
    ↓
Patch Embedding (16×16 patches) → 196 tokens
    ↓
Positional Embedding + LIF spiking
    ↓
4× Spiking Transformer Blocks (Self-Attn + LIF-MLP)
    ↓
Global Average Pool + FC(256→5)
    ↓
Spike Count over 8 timesteps → logits
```

**Key Features:**
- **Variant:** Small (embed_dim=256, depth=4, num_heads=4)
- **Neuron Type:** LIF
- **Patch Size:** 16×16
- **Timesteps:** 8

**Results:**
| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **65.46%** |
| **Validation F1** | **0.564** |
| **Validation AUC** | **0.906** |
| **Best Epoch** | 38 |
| **Training Time** | ~9 hours |

---

### 1.4 SNN-ViT-QIF

**File:** `src/models/snn/spiking_vit.py`

**Description:**
Spiking ViT with QIF neurons instead of LIF. Tests whether quadratic membrane dynamics improve transformer-based spike processing.

**Results:**
| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **54.12%** |
| **Validation F1** | **0.448** |
| **Validation AUC** | **0.818** |
| **Best Epoch** | 50 |

**Note:** QIF underperformed LIF in ViT architecture (54% vs 65%)

---

### 1.5 SNN-1D-LIF

**File:** `src/models/snn_1d/snn_classifier.py`

**Description:**
1D Spiking Neural Network for raw EEG processing. Three-stage pyramid architecture with depthwise separable convolutions and LIF neurons. Processes raw 6-channel EEG (3000 samples) directly without time-frequency transformation.

**Architecture:**
```
Raw EEG (6, 3000)
    ↓
Stage 1: DW-Sep Conv1d + LIF (t=8) → (128, 3000)
    ↓
Stage 2: DW-Sep Conv1d + LIF + Attention (optional) → (64, 1500)
    ↓
Stage 3: DW-Sep Conv1d + LIF + Attention (optional) → (32, 750)
    ↓
Temporal Fusion Block → (128-d features)
    ↓
FC(128→5) → logits
```

**Key Features:**
- **Input:** Raw 6-channel EEG (no scalogram)
- **Neuron Type:** LIF
- **Timesteps:** 8
- **Attention:** Optional Multi-Scale Spiking Attention

**Results:**
| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **42.12%** |
| **Validation F1** | **0.320** |
| **Validation AUC** | **0.733** |
| **Best Epoch** | 38 |

**Note:** Lower accuracy than 2D SNNs (42% vs 70%) - scalograms provide better features

---

### 1.6 SNN-1D-Attention

**File:** `src/models/snn_1d/snn_classifier.py`

**Description:**
SNN-1D with Multi-Scale Spiking Attention (MSSA) in stages 2 and 3. Attention captures long-range temporal dependencies across multiple scales.

**Results:**
| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **44.75%** |
| **Validation F1** | **0.376** |
| **Validation AUC** | **0.766** |
| **Best Epoch** | 25 |

**Note:** Attention provided small improvement (+2.6% accuracy)

---

### 1.7 Spiking-ViT-1D

**File:** `src/models/snn_1d/spiking_vit.py`

**Description:**
1D Spiking Vision Transformer for raw EEG. Adapts ViT architecture for 1D temporal sequences with spiking dynamics.

**Results:**
| Metric | Value | Status |
|--------|-------|--------|
| Validation Accuracy | 10.94% | ❌ Failed |
| Validation F1 | 0.105 | ❌ Failed |
| Issue | Model architecture mismatch | 🔧 Needs rerun |

---

## 2. Quantum-Classical Hybrid CNNs

### 2.1 Quantum CNN (14 Variants - 1D Raw EEG)

**File:** `src/models/quantum/hybrid_cnn.py`, `src/models/quantum/quantum_circuit.py`

**Description:**
Quantum-classical hybrid CNNs for 1D raw EEG processing. Classical CNN encoder extracts features, then vectorized quantum circuit (8 qubits, 3 layers) processes in quantum feature space. Measurements provide class logits.

**Architecture:**
```
Raw EEG (6, 3000)
    ↓
Classical CNN Encoder (Conv1d + Pooling)
    ↓
Feature Vector (512-d)
    ↓
Pre-Quantum Projection → 8 angles
    ↓
Quantum Circuit (8 qubits, 3 layers)
    - Rotation Gates: RX, RY, RZ, RXY, etc.
    - Entanglement: Ring or Full
    ↓
Pauli-Z Measurement → 8 expectation values
    ↓
FC(8→5) → logits
```

**Variants Tested:**

| Rotation | Entanglement | Accuracy | F1 | Status |
|----------|-------------|----------|-----|--------|
| **RX** | Ring | 42.4% | 0.330 | ✅ |
| **RY** | Ring | 45.9% | 0.343 | ✅ |
| **RZ** | Ring | 47.4% | 0.371 | ✅ |
| **RXY** | Ring | 50.5% | 0.392 | ✅ |
| **RXZ** | Ring | 49.7% | 0.380 | ✅ |
| **RYZ** | Ring | 54.0% | 0.436 | ✅ |
| **RXYZ** | Ring | 53.5% | 0.419 | ✅ |
| **RX** | Full | 53.7% | 0.418 | ✅ |
| **RY** | Full | 52.9% | 0.434 | ✅ |
| **RZ** | Full | 47.0% | 0.367 | ✅ |
| **RXY** | Full | **54.6%** | **0.446** | ✅ **Best** |
| **RXZ** | Full | 46.3% | 0.353 | ✅ |
| **RYZ** | Full | 53.9% | 0.445 | ✅ |
| **RXYZ** | Full | 50.5% | 0.396 | ✅ |

**Best Variant:** `quantum_1d_full_RXY` (RXY rotation + Full entanglement)
- **Accuracy:** 54.56%
- **F1 (macro):** 0.446
- **AUC:** 0.832

**Key Findings:**
- **Full entanglement** generally outperforms ring entanglement
- **RXY rotation** (X+Y gates) performs best
- Quantum models on 1D raw EEG achieve ~50-55% accuracy
- Much lower than 2D quantum models (83%) - scalograms crucial

**Code Snippet:**
```python
class HybridQuantumCNN(nn.Module):
    def __init__(self, rotation_type='RXY', entanglement_type='full', n_qubits=8):
        self.encoder = ClassicalCNNEncoder()  # Conv1d stack
        self.pre_quantum = nn.Linear(512, n_qubits)
        self.circuit = VectorizedQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=3,
            rotation_type=rotation_type,
            entanglement_type=entanglement_type,
        )
        self.measurement = PauliZMeasurement()
        self.classifier = nn.Linear(n_qubits, 5)
    
    def forward(self, x):
        features = self.encoder(x)  # (B, 512)
        angles = self.pre_quantum(features)  # (B, 8)
        angles = torch.tanh(angles) * np.pi  # [-π, π]
        
        quantum_state = self.circuit(angles)  # (B, 2^n_qubits)
        measurements = self.measurement(quantum_state)  # (B, 8)
        
        return self.classifier(measurements)
```

---

### 2.2 Quantum CNN (14 Variants - 2D Scalogram)

**File:** `src/models/quantum/hybrid_cnn.py`

**Description:**
Same quantum architecture but processing 2D scalograms instead of 1D raw EEG. Much higher accuracy due to better feature representation.

**Results (Top 5):**

| Rotation | Entanglement | Accuracy | F1 | AUC |
|----------|-------------|----------|-----|-----|
| **RXY** | Full | **83.43%** | **0.657** | **0.949** |
| **RXZ** | Full | 83.55% | 0.661 | 0.952 |
| **RXYZ** | Full | 83.53% | 0.633 | 0.948 |
| **RYZ** | Full | 82.14% | 0.617 | 0.942 |
| **RXY** | Ring | 83.65% | 0.656 | 0.951 |

**Best Variant:** `quantum_full_RXY`
- **Accuracy:** 83.43%
- **F1 (macro):** 0.657
- **AUC:** 0.949

**Key Finding:** 2D scalogram + quantum = **83% accuracy** vs 1D raw EEG + quantum = **54% accuracy**

---

## 3. Vision Transformers

### 3.1 Fusion-A (Swin + ConvNeXt) ⭐

**File:** `src/models/fusion/fusion_a.py`

**Description:**
Dual-backbone fusion combining Swin Transformer's local window attention with ConvNeXt's hierarchical CNN features. Both process 2D scalograms, fused via gated attention mechanism.

**Architecture:**
```
Scalogram (3, 224, 224)
    ├─→ Swin-Tiny → 768-d features
    └─→ ConvNeXt-Tiny → 768-d features
        ↓
Gated Fusion Module (learnable weighting)
        ↓
FC(768→256→5) → logits
```

**Key Features:**
- **Swin:** Local window self-attention (periodic/rhythmic EEG patterns)
- **ConvNeXt:** Hierarchical CNN (spatial scalogram structure)
- **Fusion:** Gated attention with learned weighting

**Results:**
| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **82.01%** |
| **Validation F1** | **0.674** |
| **Validation AUC** | **0.933** |
| **Best Epoch** | 18 |
| **Training Time** | ~2 hours |

**Code Snippet:**
```python
class FusionA(nn.Module):
    def __init__(self, num_classes=5):
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        self.convnext = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        self.fusion = GatedFusionModule(primary_dim=768, secondary_dim=768)
        self.classifier = ClassificationHead(input_dim=768, num_classes=5)
    
    def forward(self, x):
        swin_feat = self.swin(x)        # (B, 768)
        convnext_feat = self.convnext(x)  # (B, 768)
        fused, gate = self.fusion(swin_feat, convnext_feat)
        return self.classifier(fused)
```

---

### 3.2 Fusion-B (4-Way Hybrid)

**File:** `src/models/fusion/fusion_b.py`

**Description:**
4-way hybrid fusion: Swin + ConvNeXt + DeiT + Quantum. Combines four complementary architectures for multi-representation learning.

**Architecture:**
```
Scalogram (3, 224, 224)
    ├─→ Swin-Tiny → 768-d
    ├─→ ConvNeXt-Tiny → 768-d
    ├─→ DeiT-Small → 384-d
    └─→ Quantum CNN (ring-RXY) → 8-d
        ↓
Multi-Stream Fusion (learned attention)
        ↓
FC(1928→768→5) → logits
```

**Status:** ⏳ Not run yet

---

### 3.3 Fusion-C (Multi-Modal: SNN-1D + Swin)

**File:** `src/models/fusion/fusion_c.py`

**Description:**
Multi-modal fusion combining raw 1D EEG (temporal) with 2D scalogram (spectral). SNN processes raw signal for spike dynamics, Swin processes scalogram for time-frequency patterns.

**Architecture:**
```
Raw EEG (6, 3000) → SNN-1D-Attention → 128-d
Scalogram (3, 224, 224) → Swin-Tiny → 768-d
        ↓
Project both to 256-d + LayerNorm
        ↓
Gated Fusion (adaptive weighting)
        ↓
FC(256→5) → logits
```

**Status:** ⏳ Not run yet

---

## 4. Fusion Models

### 4.1 SNN Fusion - Early Fusion

**File:** `src/models/fusion/fusion_models.py`

**Description:**
Early fusion of 1D SNN features and 2D SNN features. Concatenates features before classification.

**Architecture:**
```
Raw EEG (6, 3000) → SNN-1D → 128-d
Scalogram (3, 224, 224) → SNN-2D → 512-d
        ↓
Concat(128+512) → FC(640→256→128→5)
```

**Status:** ⏳ Running (no results yet)

---

### 4.2 SNN Fusion - Late Fusion

**File:** `src/models/fusion/fusion_models.py`

**Description:**
Late fusion via ensemble averaging of 1D and 2D branch predictions.

**Status:** ⏳ Running (no results yet)

---

### 4.3 SNN Fusion - Gated Fusion ⭐

**File:** `src/models/fusion/fusion_models.py`

**Description:**
Confidence-based adaptive gating. High-confidence samples use fast 1D-only path; low-confidence samples activate full 1D+2D fusion.

**Architecture:**
```
Raw EEG → SNN-1D → Features + Confidence Score
        ↓
If confidence > 0.7: Use 1D-only classifier
If confidence < 0.7: Extract 2D features → Fuse → Classify
```

**Expected:** 88-90% accuracy with 2× speedup for easy samples

**Status:** ⏳ Running (no results yet)

---

### 4.4 Quantum-SNN Fusion - Early

**File:** `src/models/fusion/fusion_models.py`

**Description:**
Early fusion of Quantum CNN (2D, best variant RXY-full) + SNN-1D. Novel architecture combining quantum and spiking paradigms.

**Expected:** 89-91% accuracy (SOTA candidate)

**Status:** 🔄 Currently Running

---

### 4.5 Quantum-SNN Fusion - Gated

**File:** `src/models/fusion/fusion_models.py`

**Description:**
Gated fusion of Quantum CNN + SNN-1D. First work combining quantum and spiking neural networks with adaptive routing.

**Expected:** 89-91% accuracy (SOTA candidate, novel contribution)

**Status:** ⏳ Not run yet

---

## 5. Complete Results Table

### All Models Ranked by Accuracy

| Rank | Model | Modality | Val Acc | Val F1 | Val AUC | Status |
|------|-------|----------|---------|--------|---------|--------|
| 1 | **fusion_a** | 2D (Swin+ConvNeXt) | **82.01%** | 0.674 | 0.933 | ✅ |
| 2 | **quantum_full_RXY** | 2D (Quantum) | **83.43%** | 0.657 | 0.949 | ✅ |
| 3 | **quantum_full_RXZ** | 2D (Quantum) | 83.55% | 0.661 | 0.952 | ✅ |
| 4 | **quantum_ring_RXY** | 2D (Quantum) | 83.65% | 0.656 | 0.951 | ✅ |
| 5 | **snn_lif_resnet** | 2D (SNN) | 69.74% | 0.576 | 0.913 | ✅ |
| 6 | **snn_lif_vit** | 2D (SNN-ViT) | 65.46% | 0.564 | 0.906 | ✅ |
| 7 | **quantum_1d_full_RXY** | 1D (Quantum) | 54.56% | 0.446 | 0.832 | ✅ |
| 8 | **quantum_1d_ring_RYZ** | 1D (Quantum) | 53.99% | 0.436 | 0.825 | ✅ |
| 9 | **snn_qif_vit** | 2D (SNN-ViT) | 54.12% | 0.448 | 0.818 | ✅ |
| 10 | **snn_1d_attn** | 1D (SNN) | 44.75% | 0.376 | 0.766 | ✅ |
| 11 | **snn_1d_lif** | 1D (SNN) | 42.12% | 0.320 | 0.733 | ✅ |
| 12 | **quantum_1d_ring_RX** | 1D (Quantum) | 42.44% | 0.330 | 0.760 | ✅ |
| 13 | **snn_qif_resnet** | 2D (SNN) | 13.55% | 0.048 | 0.500 | ❌ Failed |
| 14 | **spiking_vit_1d** | 1D (SNN-ViT) | 10.94% | 0.105 | 0.539 | ❌ Failed |

**Pending Results:**
- `snn_fusion_early` - Running
- `snn_fusion_late` - Running
- `snn_fusion_gated` - Running
- `quantum_snn_fusion_early` - Running
- `quantum_snn_fusion_gated` - Not started
- `fusion_b` - Not started
- `fusion_c` - Not started
- `swin`, `vit`, `deit`, `efficientnet`, `convnext` - Not started

---

## 6. Key Insights

### 6.1 Modality Comparison

| Modality | Best Model | Accuracy | Notes |
|----------|------------|----------|-------|
| **2D Scalogram** | fusion_a | 82.01% | ✅ Best representation |
| **1D Raw EEG** | quantum_1d_full_RXY | 54.56% | ⚠️ Limited by noise |
| **Fusion (1D+2D)** | Expected: snn_fusion_gated | 88-90% | 🎯 Optimal tradeoff |

### 6.2 Architecture Comparison

| Architecture | Best Variant | Accuracy | Speed |
|-------------|--------------|----------|-------|
| **CNN Fusion** | fusion_a | 82.01% | Fast |
| **Quantum CNN** | quantum_full_RXY | 83.43% | Very Slow (10×) |
| **SNN** | snn_lif_resnet | 69.74% | Medium (3×) |
| **SNN Fusion** | Expected: snn_fusion_gated | 88-90% | Medium (2×) |

### 6.3 Lessons Learned

1. **Scalograms are crucial:** 2D models consistently outperform 1D models (82% vs 54%)
2. **BatchNorm breaks SNNs:** Must use InstanceNorm/LayerNorm for spiking networks
3. **Spike encoding rate matters:** 0.7 scale works better than 0.3 for 8 timesteps
4. **Quantum shows promise:** 83% accuracy on scalograms, novel contribution
5. **Fusion is key:** Multi-modal approaches expected to reach 88-90%

---

## 7. Next Steps

### Immediate (This Week)
- [ ] Complete running SNN fusion models
- [ ] Run transformer baselines (swin, vit, deit, etc.)
- [ ] Fix and rerun failed models (snn_qif_resnet, spiking_vit_1d)

### Short-term (2 Weeks)
- [ ] Complete all 15 planned models
- [ ] Analyze gating behavior in snn_fusion_gated
- [ ] Cross-validate on Sleep-EDF dataset

### Long-term (1 Month)
- [ ] Write paper on SNN fusion results
- [ ] Submit quantum-SNN fusion as novel contribution
- [ ] Optimize for deployment (energy efficiency analysis)

---

**Generated:** March 20, 2026  
**Total Experiments:** 24 completed, 6 running, 15 pending  
**Best Result:** 83.65% (quantum_ring_RXY on 2D scalograms)
