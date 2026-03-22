# EEG Sleep Staging Project Report

**Prepared for:** Mentor review  
**Date:** March 23, 2026  
**Primary dataset discussed here:** BOAS / OpenNeuro `ds005555`  
**Primary latest result source:** `wandb_export_2026-03-22T13_09_19.344+05_30.csv`  
**Scope of this report:** historical 2D experiments, historical 1D experiments, and the latest improved runs and fusion experiments

---

## 1. Executive Summary

This project is a broad EEG sleep staging benchmark built around a 5-class BOAS dataset setup:

- Classes: Wake, N1, N2, N3, REM
- Representations explored:
  - **2D scalograms** from EEG epochs
  - **1D raw EEG signals**
  - **multi-branch fusion** of raw and scalogram features
- Model families explored:
  - Spiking Neural Networks (SNNs)
  - Quantum-classical hybrid models
  - Vision Transformer and CNN baselines
  - Multiple fusion architectures

The work happened in **three major phases**:

1. **Phase A: First 2D runs**
   - Initial focus on scalogram-based 2D models
   - Strong early performance from 2D quantum and fusion models

2. **Phase B: 1D/raw-signal expansion**
   - Added 1D SNN and 1D quantum variants
   - Early raw-signal results were clearly weaker than 2D

3. **Phase C: Latest improved paper runs**
   - Training refinements, cleaner model selection, more systematic comparisons
   - Major improvement in 1D models
   - Fusion-B became the best finished run in the latest export

**Most important high-level conclusion so far:**

- The project consistently shows that **2D spectral representations are stronger overall than raw 1D alone**
- But the **latest improved 1D models are now much more competitive**
- Fusion remains the most promising direction, especially:
  - `fusion_a`: dual-backbone 2D fusion
  - `fusion_b`: 4-way hybrid 2D fusion
  - `fusion_c`: multi-modal 1D + 2D fusion, still running / not finalized in the March 22 export

---

## 2. Project Objective

The main goal of the project has been to build a **comprehensive EEG sleep staging benchmark** across several different modeling paradigms, not just to optimize a single model.

The project tries to answer the following questions:

- How strong are **standard 2D image-style models** on EEG scalograms?
- Can **spiking neural networks** model EEG effectively?
- Can **quantum-classical hybrids** provide useful gains?
- How much information is lost when using raw 1D EEG instead of 2D time-frequency transforms?
- Can **fusion architectures** combine temporal and spectral information in a useful way?

This makes the project useful not just as a single benchmark, but also as a **comparative research framework**.

---

## 3. Dataset and Experimental Setup

### 3.1 Primary Dataset

The main training and comparison work in this report is based on:

- **BOAS / OpenNeuro ds005555**
- EEG sleep staging
- 5 classes: `W`, `N1`, `N2`, `N3`, `REM`
- Epoch-based classification

### 3.2 Data Representations Used

Two main forms of data were used:

#### A. 2D Scalogram Representation

- Raw EEG epochs are converted to time-frequency images
- These are then treated as image-like inputs of size roughly `3 x 224 x 224`
- Used by:
  - 2D SNNs
  - quantum 2D models
  - Swin / ViT / DeiT / ConvNeXt / EfficientNet
  - Fusion-A and Fusion-B

#### B. 1D Raw EEG Representation

- Raw EEG signals are used directly as temporal sequences
- Typical shape in the codebase: `6 x 3000`
- Used by:
  - 1D SNNs
  - 1D quantum models
  - TCANet-style branch
  - raw branch inside fusion models

### 3.3 Evaluation Caveat

The **latest W&B export is the best source of truth for the latest paper-run phase**, but it is still based on `fold0` runs in the export table, not the full 5-fold cross-validation summary yet.

So this report separates:

- **historical bests from earlier phases**
- **latest finished March 22 paper runs**

This is important because some latest runs are improved in stability or methodology, but not always numerically better than the historical peak values.

---

## 4. Phase A - First Set of Runs on 2D Data

The project started with a strong emphasis on **2D scalogram-based learning**. This was the most natural starting point because many powerful computer vision backbones can be reused once EEG is transformed into a time-frequency image.

### 4.1 2D Model Families Tried

The 2D phase included:

- 2D SNN ResNet
- 2D SNN ViT
- 2D quantum hybrid CNNs
- transformer/CNN baselines such as Swin, ViT, DeiT, EfficientNet, ConvNeXt
- 2D fusion models

### 4.2 Main Historical 2D Findings

From the earlier BOAS documentation and result summaries, the strongest historical 2D results were:

| Model | Representation | Historical Result |
|------|------|------|
| `quantum_ring_RXY` | 2D scalogram | **83.65% acc** |
| `quantum_full_RXY` | 2D scalogram | **83.43% acc** |
| `fusion_a` | 2D dual-backbone fusion | **82.01% acc** |
| `snn_lif_resnet` | 2D SNN | **69.74% acc** |
| `snn_lif_vit` | 2D SNN-ViT | **65.46% acc** |

### 4.3 What This Told Us Early

The first wave of experiments made several things clear:

- **2D was strong immediately**
- The **best results came from either quantum 2D or fusion 2D**
- Classical and hybrid 2D models were easier to optimize than raw 1D models
- The SNNs worked, but they lagged behind the strongest non-spiking 2D approaches

### 4.4 Interpretation

This phase suggested that the sleep staging signal contains a lot of useful information in its **time-frequency structure**, and that converting EEG into scalograms gave the models a more accessible and learnable input space.

This was the point where the project established:

- a strong 2D baseline
- a working 2D quantum benchmark
- evidence that fusion could outperform single-branch models

---

## 5. Phase B - Next Set of Runs on 1D Raw EEG

After the 2D work, the project expanded to direct **raw EEG modeling**. This phase aimed to answer whether the model could learn sleep stages without the time-frequency transform.

### 5.1 1D Model Families Tried

The 1D phase included:

- `snn_1d_lif`
- `snn_1d_attn`
- `spiking_vit_1d`
- a full 14-variant set of **1D quantum models**

### 5.2 Historical 1D Results Before Major Improvements

Earlier documentation recorded the following rough results:

| Model | Historical Result |
|------|------|
| `snn_1d_attn` | **44.75% acc** |
| `snn_1d_lif` | **42.12% acc** |
| `quantum_1d_full_RXY` | **54.56% acc** |
| `quantum_1d_ring_RYZ` | **53.99% acc** |
| `spiking_vit_1d` | **10.94% acc** failed |

### 5.3 What This Told Us

At this stage, the 1D story was mixed:

- Raw EEG alone was clearly harder than 2D scalograms
- 1D SNNs initially underperformed badly relative to 2D
- 1D quantum models were better than early 1D SNNs, but still clearly below strong 2D models
- The results suggested that **raw temporal information was useful**, but difficult to exploit directly without stronger architectural and training changes

### 5.4 Why the 1D Phase Was Still Important

Even though the early 1D numbers were weaker, this phase was essential because it established:

- a raw-signal benchmark
- the gap between 1D and 2D
- the motivation for **multi-modal fusion**
- the need for training improvements and architecture fixes

Without this phase, there would have been no strong reason to invest in raw+spectral fusion.

---

## 6. Phase C - Latest Improved Runs and Changes

The latest paper-oriented run series, captured in the March 22 W&B export, reflects a newer stage of the project:

- better training settings
- cleaner experiment selection
- stronger 1D runs
- fusion experiments that are much closer to publication-style reporting

### 6.1 Latest Finished Results from March 22 W&B Export

At the timestamp of the export, the finished runs ranked as follows:

| Rank | Model | Modality | Accuracy | F1 Macro | AUC | Best Epoch |
|------|------|------|------|------|------|------|
| 1 | `fusion_b` | 2D 4-way fusion | **82.44%** | 0.679 | 0.935 | 22 |
| 2 | `fusion_a` | 2D dual fusion | **80.79%** | 0.684 | 0.947 | 11 |
| 3 | `convnext` | 2D CNN baseline | **80.50%** | 0.658 | 0.937 | 31 |
| 4 | `snn_1d_attn` | 1D raw EEG | **75.76%** | 0.638 | 0.933 | 23 |
| 5 | `quantum_ring_RXY` | 2D quantum | **75.05%** | 0.635 | 0.944 | 32 |
| 6 | `quantum_full_RXY` | 2D quantum | **73.79%** | 0.623 | 0.940 | 18 |
| 7 | `snn_1d_lif` | 1D raw EEG | **73.73%** | 0.616 | 0.931 | 35 |
| 8 | `swin` | 2D transformer | **70.65%** | 0.600 | 0.928 | 3 |
| 9 | `snn_lif_resnet` | 2D SNN | **67.47%** | 0.569 | 0.902 | 16 |
| 10 | `snn_lif_vit` | 2D SNN-ViT | **60.57%** | 0.516 | 0.868 | 9 |

### 6.2 Runs Still In Progress at Export Time

The March 22 export also showed:

- `fusion_c` was still running
- a rerun of `fusion_a` was still running and had reached about **81.08%** validation accuracy at epoch 30

This means the March 22 export is a **snapshot**, not the final endpoint of all current experiments.

### 6.3 Most Important Change in the Latest Phase

The biggest positive change in the latest phase is the raw-signal branch:

| Model | Earlier Result | Latest Result | Gain |
|------|------|------|------|
| `snn_1d_lif` | 42.12% | **73.73%** | **+31.61 points** |
| `snn_1d_attn` | 44.75% | **75.76%** | **+31.01 points** |

This is a major improvement and changes the interpretation of the project:

- earlier: raw 1D seemed weak
- now: raw 1D is clearly viable
- this makes fusion even more promising, because the raw branch is no longer too weak to contribute

### 6.4 Important Nuance About 2D Quantum Models

The latest paper-run 2D quantum results were lower than the strongest historical numbers:

| Model | Historical Peak | Latest March 22 Result |
|------|------|------|
| `quantum_ring_RXY` | 83.65% | 75.05% |
| `quantum_full_RXY` | 83.43% | 73.79% |

This does **not automatically mean the models got worse** in a scientific sense. It more likely indicates one or more of the following:

- changed training setup
- changed subset or split behavior
- cleaner evaluation
- differences between older broad experiment runs and the newer paper-oriented run series

For mentor discussion, this should be framed as:

> The latest paper-style run series improved some branches a lot, especially 1D raw EEG, but some earlier 2D quantum peaks have not yet been fully matched under the newer run configuration. This suggests the need for controlled reruns and standardization of evaluation before final publication claims.

---

## 7. Detailed Model Families Tried So Far

This section summarizes all major model families explored in the project.

### 7.1 2D Spiking Neural Networks

Models tried:

- `snn_lif_resnet`
- `snn_qif_resnet`
- `snn_lif_vit`
- `snn_qif_vit`

Purpose:

- Test biologically inspired spike-based processing on time-frequency EEG representations
- Compare LIF and QIF neuron dynamics
- Compare CNN-style and transformer-style spiking backbones

Findings:

- LIF variants were clearly stronger than failed or unstable alternatives
- `snn_lif_resnet` was the best 2D SNN
- 2D SNNs worked, but were not the top overall family

### 7.2 1D Spiking Neural Networks

Models tried:

- `snn_1d_lif`
- `snn_1d_attn`
- `spiking_vit_1d`

Purpose:

- Directly model raw EEG temporal structure
- Avoid relying entirely on handcrafted spectral transforms
- Test whether attention helps raw spiking models

Findings:

- The raw 1D SNNs improved dramatically in the latest phase
- Attention gave a consistent gain over plain 1D LIF
- `spiking_vit_1d` was not successful in its earlier form

### 7.3 2D Quantum-Classical Hybrid Models

Models tried:

- 14 rotation/entanglement combinations:
  - rotations: `RX`, `RY`, `RZ`, `RXY`, `RXZ`, `RYZ`, `RXYZ`
  - entanglement: `ring`, `full`

Purpose:

- Compress 2D scalogram features into a quantum feature space
- Test whether structured quantum transformations help EEG discrimination

Findings:

- `RXY` consistently emerged as one of the strongest rotation settings
- The quantum models were among the strongest historical performers
- In the latest phase they remained competitive, but not dominant

### 7.4 1D Quantum Models

Models tried:

- another full 14-variant sweep on raw EEG

Purpose:

- Test whether quantum processing can compensate for the difficulty of raw 1D EEG

Findings:

- 1D quantum performed better than early 1D SNNs
- But after the latest improvements, the 1D SNN branch became more competitive
- This is now a more balanced comparison than before

### 7.5 Classical 2D Baselines

Models tried:

- `swin`
- `vit`
- `deit`
- `efficientnet`
- `convnext`

Purpose:

- Establish strong non-spiking, non-quantum baselines
- See whether EEG scalograms can be solved best with standard CV architectures

Findings from latest completed runs:

- `convnext` performed very strongly at **80.50%**
- `swin` was weaker in the latest run
- the classical 2D baseline family remains a very important reference point

### 7.6 TCANet

TCANet was added later as a clean 1D/raw model direction and has now been integrated with a real quantum bottleneck path in the codebase.

At the time of the March 22 export:

- TCANet was integrated into the project pipeline
- but it was **not yet part of the latest W&B result table**

So it should be described as:

> a newly integrated model direction prepared for the next experiment wave rather than a completed BOAS benchmark result in the current report.

---

## 8. Fusion Models and Fusion Strategies Tried So Far

Fusion is one of the most important parts of the project, because the work has repeatedly shown that raw temporal information and spectral image information are complementary.

### 8.1 Why Fusion Was Added

The fusion models were motivated by a clear pattern:

- 2D scalograms were strong
- raw 1D initially lagged
- later, raw 1D improved a lot

This naturally led to the idea that:

- 1D captures **temporal dynamics**
- 2D captures **spectral/time-frequency structure**
- fusion may combine both advantages

### 8.2 Fusion-A: Dual 2D Fusion

**Model:** `fusion_a`

Structure:

- Swin branch on scalogram
- ConvNeXt branch on scalogram
- gated fusion of the two feature streams

Interpretation:

- This is a **2D-only fusion**
- It combines two different visual inductive biases:
  - window-based transformer attention
  - hierarchical convolutional features

Results:

- Historical result: **82.01%**
- Latest finished March 22 result: **80.79%**
- Still one of the strongest and cleanest models in the project

### 8.3 Fusion-B: 4-Way Hybrid Fusion

**Model:** `fusion_b`

Structure:

- Swin branch
- ConvNeXt branch
- DeiT branch
- quantum branch
- multi-stream fusion head

Interpretation:

- This is the most ambitious completed 2D fusion model so far
- It combines:
  - local window transformer features
  - hierarchical CNN features
  - global transformer features
  - quantum-compressed feature representations

Latest result:

- **82.44% accuracy**
- **0.679 macro-F1**
- best finished model in the March 22 export

This is currently the best finished paper-run model in the latest W&B export.

### 8.4 Fusion-C: Multi-Modal 1D + 2D Fusion

**Model:** `fusion_c`

Structure:

- 1D raw EEG branch using SNN-1D attention features
- 2D scalogram branch using a 2D backbone
- gated fusion head for joint classification

Original idea:

- raw branch captures temporal/signal-level patterns
- 2D branch captures spectral structure
- gating decides how much each representation should contribute

Important engineering note:

- In documentation this was originally described with Swin
- In the actual code it was changed to a lighter 2D backbone for speed

Status at the March 22 export:

- still running
- not yet available as a finalized comparison point in the export

Why it matters:

- it is the most direct test of the core project hypothesis:
  - **temporal + spectral fusion should outperform either alone**

### 8.5 SNN Fusion Variants

Additional fusion ideas already designed in the repo:

- `snn_fusion_early`
- `snn_fusion_late`
- `snn_fusion_gated`

These represent different fusion philosophies:

#### Early Fusion

- combine features from the two modalities before final classification
- strongest information mixing
- potentially highest accuracy

#### Late Fusion

- let each branch make its own prediction
- combine predictions later
- simpler and more interpretable

#### Gated Fusion

- adaptive routing based on confidence or learned gating
- aims for a better efficiency/accuracy trade-off
- conceptually one of the most interesting designs in the repo

### 8.6 Quantum-SNN Fusion Variants

Further fusion ideas include:

- `quantum_snn_fusion_early`
- `quantum_snn_fusion_gated`

These are important because they try to combine:

- raw temporal spike-based modeling
- spectral quantum-classical processing

This is one of the most novel directions in the project, although it is still more of an active research branch than a finalized completed benchmark result.

---

## 9. Chronological Story of the Project

The project progression can be summarized cleanly as follows:

### Stage 1: Build strong 2D baselines

- Use EEG scalograms
- Benchmark SNNs, quantum models, and vision models
- Outcome:
  - very strong 2D performance
  - quantum and 2D fusion emerge as top contenders

### Stage 2: Add raw 1D modeling

- Introduce 1D SNN and 1D quantum approaches
- Outcome:
  - raw EEG is harder
  - early results lag behind 2D
  - motivates model and training improvements

### Stage 3: Improve training and architecture pipeline

- 1D branch becomes dramatically stronger
- new paper-run results show 1D SNNs now near mid/high-70s
- Outcome:
  - raw 1D becomes much more credible
  - fusion becomes more scientifically meaningful

### Stage 4: Build fusion architectures

- dual-backbone fusion
- 4-way hybrid fusion
- multimodal raw+scalogram fusion
- SNN fusion and quantum-SNN fusion concepts
- Outcome:
  - `fusion_b` becomes best finished latest run
  - `fusion_c` remains a key unfinished but important direction

---

## 10. Main Findings for Mentor Discussion

The clearest takeaways to present are:

### 10.1 2D Scalograms Are Still the Strongest General Representation

Across the project, 2D models consistently produced the strongest single-model and fusion results.

### 10.2 1D Raw EEG Was Weak Early, but Improved Dramatically

The latest phase changed the raw EEG story significantly:

- `snn_1d_lif`: 42.12% -> 73.73%
- `snn_1d_attn`: 44.75% -> 75.76%

This is one of the strongest positive developments in the project.

### 10.3 Fusion Is the Most Promising Final Direction

Fusion models are now the most exciting part of the project because they:

- bring together different information sources
- outperform or match strong single branches
- provide a clean publication story

### 10.4 Latest Best Finished Result

As of the March 22 export:

- **best finished latest run = `fusion_b`**
- accuracy = **82.44%**
- macro-F1 = **0.679**
- AUC = **0.935**

### 10.5 Best Historical Results and Latest Results Are Not Yet Fully Aligned

Some older 2D quantum runs were numerically stronger than the latest paper-run quantum results.

This means the next important step is not just adding more models, but:

- standardizing the evaluation protocol
- rerunning key candidates under one consistent setup
- making the final benchmark directly comparable

---

## 11. Recommended Next Steps

For the immediate next iteration, the most useful work would be:

1. Finalize `fusion_c` and make it practical to train
2. Standardize reporting across older and newer runs
3. Rerun top candidates under a unified setup:
   - `fusion_b`
   - `fusion_a`
   - `convnext`
   - `snn_1d_attn`
   - `quantum_ring_RXY`
   - `quantum_full_RXY`
4. Add TCANet to the same benchmark table once trained
5. Extend from BOAS to Sleep-EDF for generalization

---

## 12. Final Summary

This project has evolved from a broad exploratory benchmark into a much more structured EEG sleep staging research pipeline.

The major milestones so far are:

- strong 2D scalogram-based baselines
- full 1D raw EEG expansion
- major 1D improvement in the latest runs
- multiple fusion architectures, with `fusion_b` currently the best finished latest run
- `fusion_c` and multimodal fusion remaining a major next milestone

If summarized in one sentence:

> The project shows that 2D spectral modeling remains the strongest overall foundation, raw 1D modeling has improved enough to become genuinely competitive, and fusion of these two views is the most promising route toward the final best-performing EEG sleep staging system.
