# EEG Sleep Staging Project Report

**Prepared for:** Mentor review  
**Date:** March 23, 2026  
**Primary dataset:** BOAS / OpenNeuro `ds005555`  
**Latest result source used for this revision:** `wandb_export_2026-03-23T00_46_06.065+05_30.csv`  
**Scope:** all major BOAS experiments so far, including early 2D runs, 1D expansion, latest improved runs, all fusion strategies, and architectures implemented in the repo even if not yet fully benchmarked

---

## 1. Executive Summary

This project is a broad EEG sleep staging benchmark built on a 5-class BOAS setup:

- Classes: `W`, `N1`, `N2`, `N3`, `REM`
- Modalities explored:
  - **2D scalograms**
  - **1D raw EEG**
  - **multi-branch fusion of 1D and 2D**
- Model families explored:
  - classical CNN / transformer backbones
  - spiking neural networks
  - quantum-classical hybrids
  - multimodal fusion systems
  - TCANet

The project has progressed in three broad waves:

1. **Wave A: 2D-first benchmarking**
   - scalogram-based image models
   - strongest early results came from 2D quantum and 2D fusion branches

2. **Wave B: 1D/raw EEG expansion**
   - raw temporal models, including 1D SNNs and 1D quantum models
   - early 1D results were weaker and less stable than 2D

3. **Wave C: newer improved paper-style runs**
   - much stronger 1D performance
   - stronger controlled comparisons among selected candidates
   - `fusion_b` became the best finished run in the latest paper-run group
   - `fusion_c` entered active training as the main multimodal raw+spectral fusion candidate

The clearest high-level message is:

> 2D spectral modeling has been the strongest foundation overall, 1D raw EEG improved dramatically in the later runs, and fusion remains the most promising final direction.

---

## 2. Project Objective

The goal of the project is not just to optimize one model, but to build a **comparative research framework** for EEG sleep staging.

The project is effectively asking:

- How far can we go with **2D time-frequency EEG images**?
- How much can be learned directly from **raw 1D EEG**?
- Are **spiking models** useful for EEG?
- Are **quantum feature bottlenecks** useful for EEG?
- Can **fusion architectures** combine temporal and spectral information better than either alone?

This makes the repo useful both as:

- a benchmark suite
- a research prototype platform
- a staging ground for publication-oriented comparisons

---

## 3. Dataset and Task Setup

### 3.1 Main Dataset

The primary completed results in this report come from:

- **BOAS / OpenNeuro `ds005555`**
- 5-class sleep stage classification
- epoch-level EEG classification

### 3.2 Representations Used

#### A. 2D Scalograms

- raw EEG epochs are transformed into time-frequency images
- typical input shape: about `3 x 224 x 224`
- used by:
  - 2D SNNs
  - transformer/CNN baselines
  - 2D quantum models
  - `fusion_a`
  - `fusion_b`

#### B. 1D Raw EEG

- raw EEG is used as a temporal signal directly
- typical input shape in the current codebase: `6 x 3000`
- used by:
  - 1D SNNs
  - 1D quantum models
  - TCANet
  - the raw branch of multimodal fusion models

#### C. Multi-Modal 1D + 2D

- raw EEG and scalograms are processed in parallel
- used by:
  - `fusion_c`
  - SNN fusion variants
  - quantum-SNN fusion variants

### 3.3 Evaluation Caveat

The latest CSV is the best available summary of actual W&B-tracked history, but it still mixes:

- early exploratory runs
- later improved paper-style reruns
- in-progress experiments
- code paths that exist but do not yet have a clean benchmark result

So this report explicitly separates:

- **historical peaks**
- **latest rerun results**
- **in-progress**
- **implemented but not yet cleanly benchmarked**

---

## 4. Chronological Development of the Project

### Stage 1: 2D Scalogram Foundation

The project started with 2D scalogram modeling because it allowed direct reuse of strong computer vision backbones.

What was tried:

- 2D spiking ResNet and ViT variants
- transformer baselines
- CNN baselines
- full 2D quantum sweep

What this established:

- 2D models were immediately strong
- quantum 2D models became top performers early
- standard CV backbones were very competitive
- this phase created a strong benchmark baseline

### Stage 2: Raw 1D EEG Expansion

The second phase tested whether direct temporal modeling could work without converting EEG into images.

What was tried:

- `snn_1d_lif`
- `snn_1d_attn`
- `spiking_vit_1d`
- full 14-model 1D quantum sweep

What this established:

- raw EEG was harder to optimize
- early 1D performance was much weaker than 2D
- 1D quantum models were often better than early 1D SNNs
- this phase created the motivation for raw+spectral fusion

### Stage 3: Improved Paper-Run Phase

The later BOAS runs were more selective and more publication-oriented.

What changed:

- stronger training setup
- cleaner experiment selection
- more mature 1D models
- more serious fusion comparisons
- TCANet entered the system as a new raw-EEG direction

What this changed scientifically:

- 1D SNNs improved from low/mid-40% accuracy into the mid/high-70s
- fusion became more meaningful because the 1D branch was no longer too weak
- the project shifted from “2D clearly wins” to “fusion is the real final target”

---

## 5. All Architecture Families Tried or Implemented

This section is the full architecture inventory as of the current repo and the March 23 CSV.

### 5.1 2D Spiking Neural Networks

Implemented / tried:

- `snn_lif_resnet`
- `snn_qif_resnet`
- `snn_lif_vit`
- `snn_qif_vit`

Purpose:

- apply biologically inspired spiking dynamics to scalogram images
- compare ResNet-style and transformer-style spiking backbones
- compare LIF vs QIF neuron behavior

Observed pattern:

- LIF variants consistently outperformed QIF variants
- `snn_lif_resnet` became the best 2D spiking CNN branch
- `snn_lif_vit` was sometimes competitive but less consistent
- QIF branches underperformed and were often unstable

### 5.2 Classical 2D Baselines

Implemented / tried:

- `swin`
- `vit`
- `deit`
- `efficientnet`
- `convnext`

Purpose:

- establish strong non-spiking, non-quantum baselines
- test whether EEG scalograms benefit from standard vision backbones

Observed pattern:

- this family is one of the strongest overall baselines in the project
- `swin`, `vit`, `deit`, `efficientnet`, and `convnext` all produced competitive 2D results
- in early runs, several of these models were above 83%
- in the latest improved paper-run subset, `convnext` was the strongest finished classical baseline

### 5.3 2D Quantum-Classical Hybrid Models

Implemented / tried:

- 14 total variants from:
  - entanglement: `ring`, `full`
  - rotations: `RX`, `RY`, `RZ`, `RXY`, `RXZ`, `RYZ`, `RXYZ`

Models:

- `quantum_ring_RX`
- `quantum_ring_RY`
- `quantum_ring_RZ`
- `quantum_ring_RXY`
- `quantum_ring_RXZ`
- `quantum_ring_RYZ`
- `quantum_ring_RXYZ`
- `quantum_full_RX`
- `quantum_full_RY`
- `quantum_full_RZ`
- `quantum_full_RXY`
- `quantum_full_RXZ`
- `quantum_full_RYZ`
- `quantum_full_RXYZ`

Purpose:

- compress 2D features through quantum-inspired bottlenecks
- test entanglement and rotation strategy choices

Observed pattern:

- this family produced some of the strongest historical results in the whole project
- `RXY`, `RXZ`, and `RXYZ` settings were especially strong historically
- later paper-style reruns used the top-performing `RXY` variants for focused comparison

### 5.4 1D Spiking Neural Networks

Implemented / tried:

- `snn_1d_lif`
- `snn_1d_attn`
- `spiking_vit_1d`

Purpose:

- test direct temporal modeling from raw EEG
- see whether attention improves spiking temporal modeling
- test whether a ViT-style 1D spiking architecture can work on raw signals

Observed pattern:

- `snn_1d_attn` became the best raw 1D spiking branch
- `snn_1d_lif` also improved dramatically in later runs
- `spiking_vit_1d` remained the weakest and least successful 1D family

### 5.5 1D Quantum Models

Implemented / tried:

- another full 14-variant sweep:
  - `quantum_1d_ring_RX`
  - `quantum_1d_ring_RY`
  - `quantum_1d_ring_RZ`
  - `quantum_1d_ring_RXY`
  - `quantum_1d_ring_RXZ`
  - `quantum_1d_ring_RYZ`
  - `quantum_1d_ring_RXYZ`
  - `quantum_1d_full_RX`
  - `quantum_1d_full_RY`
  - `quantum_1d_full_RZ`
  - `quantum_1d_full_RXY`
  - `quantum_1d_full_RXZ`
  - `quantum_1d_full_RYZ`
  - `quantum_1d_full_RXYZ`

Purpose:

- test whether quantum bottlenecks can help raw EEG more than standard temporal models

Observed pattern:

- this family was very useful in the early 1D phase
- the best 1D quantum variants reached the mid-50% range
- before the later SNN improvements, these models were among the strongest raw-signal results

### 5.6 TCANet

Implemented / tried:

- `tcanet`

Role:

- a specialized raw-EEG temporal model direction
- now integrated into the project pipeline
- later modified to support a true quantum bottleneck path in the repo

Observed pattern from W&B:

- TCANet now appears in the latest CSV
- the recorded BOAS run is currently weak relative to the best 1D SNNs and 2D models
- it should be treated as an active architecture branch, not yet a top benchmark performer

### 5.7 Fusion Models

Implemented / tried:

- `fusion_a`
- `fusion_b`
- `fusion_c`
- `snn_fusion_early`
- `snn_fusion_late`
- `snn_fusion_gated`
- `quantum_snn_fusion_early`
- `quantum_snn_fusion_gated`

This family is described in detail in Section 9 because fusion is central to the project story.

---

## 6. Phase A Results: First 2D Runs

The first major strong results in the project came from 2D scalogram-based experiments.

### 6.1 Historical Strong 2D Results

From the earlier BOAS runs in W&B, the strongest historical 2D values include:

| Model | Best Historical Accuracy | Notes |
|------|------:|------|
| `swin` | **84.62%** | strongest historical classical transformer baseline |
| `efficientnet` | **84.02%** | strong compact CNN baseline |
| `quantum_ring_RXY` | **83.98%** | one of the best historical quantum runs |
| `quantum_ring_RXZ` | **83.71%** | very strong quantum variant |
| `quantum_full_RXZ` | **83.55%** | another top quantum variant |
| `quantum_full_RXYZ` | **83.53%** | strong quantum variant |
| `fusion_a` | **82.01%** | early strong fusion result |
| `convnext` | **83.11%** | strong classical CNN baseline |
| `vit` | **82.98%** | strong transformer baseline |
| `deit` | **83.28%** | strong transformer baseline |

### 6.2 What the First 2D Phase Showed

- 2D representations were strong immediately
- the best models were either:
  - quantum 2D models
  - strong classical 2D baselines
  - fusion over multiple 2D branches
- 2D SNNs were workable, but not top-of-table

This phase established the main baseline that later 1D models had to catch up to.

---

## 7. Phase B Results: 1D Raw EEG Expansion

The next major direction was direct raw EEG modeling.

### 7.1 Early 1D Results

Representative early 1D runs from W&B:

| Model | Early Accuracy | Early Macro-F1 | Interpretation |
|------|------:|------:|------|
| `snn_1d_lif` | **42.74%** | 0.320 | weak early raw-SNN baseline |
| `snn_1d_attn` | **46.68%** | 0.377 | attention helped somewhat |
| `spiking_vit_1d` | **20.42%** | 0.125 | poor final result in mature tracked run |
| `quantum_1d_full_RXY` | **54.56%** | 0.446 | one of the best 1D quantum runs |
| `quantum_1d_ring_RYZ` | **53.99%** | 0.436 | another top 1D quantum variant |

There were also some very early exploratory 1D runs with strange class behavior and poor macro-F1 despite higher raw accuracy. Those should be treated as unstable exploratory runs rather than final baselines.

### 7.2 Main Interpretation of the 1D Phase

- raw EEG was clearly harder than 2D scalograms
- 1D quantum models initially looked more competitive than 1D SNNs
- `spiking_vit_1d` did not become a strong direction
- this phase justified later work on:
  - better training
  - better 1D SNN design
  - multimodal fusion

---

## 8. Phase C Results: Latest Improved Runs

The latest focused rerun group is the one most relevant for current reporting.

### 8.1 Latest Finished Rerun Results

These are the finished later paper-style runs from the latest W&B CSV:

| Rank | Model | Modality | Best Accuracy | Best Macro-F1 | Best AUC | Notes |
|------|------|------|------:|------:|------:|------|
| 1 | `fusion_b` | 2D fusion | **82.44%** | 0.679 | 0.935 | best finished latest run |
| 2 | `fusion_a` | 2D fusion | **80.79%** | 0.684 | 0.947 | strong, stable fusion rerun |
| 3 | `convnext` | 2D | **80.50%** | 0.658 | 0.937 | strongest finished classical rerun |
| 4 | `snn_1d_attn` | 1D | **75.76%** | 0.638 | 0.933 | strongest finished raw 1D rerun |
| 5 | `quantum_ring_RXY` | 2D quantum | **75.05%** | 0.635 | 0.944 | focused rerun of top quantum family |
| 6 | `quantum_full_RXY` | 2D quantum | **73.79%** | 0.623 | 0.940 | focused rerun of top quantum family |
| 7 | `snn_1d_lif` | 1D | **73.73%** | 0.616 | 0.931 | major improvement over earlier runs |
| 8 | `swin` | 2D | **70.65%** | 0.600 | 0.928 | lower than historical Swin peak |
| 9 | `snn_lif_resnet` | 2D SNN | **67.47%** | 0.569 | 0.902 | strongest finished 2D spiking rerun |
| 10 | `snn_lif_vit` | 2D SNN | **60.57%** | 0.516 | 0.868 | weaker than ResNet-based spiking path |
| 11 | `tcanet` | 1D | **52.22%** | 0.195 | 0.487 | integrated but not yet competitive |

### 8.2 Main Positive Development

The most important progress in the latest phase is the raw 1D branch:

| Model | Earlier Accuracy | Latest Accuracy | Gain |
|------|------:|------:|------:|
| `snn_1d_lif` | 42.74% | **73.73%** | **+30.99** |
| `snn_1d_attn` | 46.68% | **75.76%** | **+28.88** |

This is one of the central scientific findings of the whole project.

### 8.3 Important Nuance on 2D Quantum and Classical Models

Some later focused reruns are numerically below earlier historical peaks.

Examples:

| Model | Historical Peak | Latest Focused Rerun |
|------|------:|------:|
| `swin` | 84.62% | 70.65% |
| `quantum_ring_RXY` | 83.98% | 75.05% |
| `quantum_full_RXY` | 83.43% | 73.79% |
| `convnext` | 83.11% | 80.50% |

This should not be framed as “the models got worse.” More likely it reflects:

- different run configuration
- better controlled reruns
- changed subset or split behavior
- paper-style filtering of the model list

The mentor-safe interpretation is:

> the newer reruns made the comparison cleaner and greatly improved the 1D branch, but the project still needs standardized reruns before comparing historical bests and later paper-style results as if they were directly identical.

---

## 9. Fusion Models and Fusion Strategies

Fusion is one of the most important parts of this project because it is where the core research hypothesis becomes strongest.

### 9.1 Why Fusion Became Important

The project repeatedly showed:

- 2D scalograms carry strong spectral structure
- raw 1D EEG carries useful temporal information
- 1D was initially weak, but later became much stronger

That naturally leads to the idea that:

- 1D and 2D are complementary
- the best final model may be a fusion model rather than a single branch

### 9.2 `fusion_a`: Dual 2D Fusion

**Model:** `fusion_a`

Structure:

- Swin branch
- ConvNeXt branch
- gated fusion head

Interpretation:

- this is a **2D-only fusion**
- it mixes transformer-style and convolutional 2D features

Results:

- historical run: **82.01%**
- latest focused rerun: **80.79%**
- consistently one of the cleanest and strongest models in the project

Architecture summary:

- Input: 2D EEG scalogram
- Branch 1: Swin extracts transformer-style windowed spatial features
- Branch 2: ConvNeXt extracts hierarchical convolutional features
- Fusion: a gated fusion module learns how much each branch should contribute
- Head: fused representation goes to the final classifier

Why it is strong:

- Swin and ConvNeXt learn different but complementary views of the same scalogram
- the gate allows the model to adaptively emphasize the more useful branch

### 9.3 `fusion_b`: 4-Way 2D Hybrid Fusion

**Model:** `fusion_b`

Structure:

- Swin branch
- ConvNeXt branch
- DeiT branch
- quantum branch
- multi-stream fusion head

Interpretation:

- this is the most ambitious completed **2D-only fusion** model
- it combines several distinct feature types into one head

Result:

- **82.44% accuracy**
- **0.679 macro-F1**
- best finished latest rerun in the CSV

Architecture summary:

- Input: 2D EEG scalogram
- Branch 1: Swin for windowed transformer features
- Branch 2: ConvNeXt for hierarchical CNN features
- Branch 3: DeiT for compact global transformer features
- Branch 4: quantum branch for quantum-compressed representation learning
- Fusion: a multi-stream fusion head combines all four embeddings before classification

Why it is strong:

- it is the richest completed ensemble of distinct inductive biases in the project
- instead of trusting one representation style, it combines several strong 2D experts

### 9.4 `fusion_c`: Multi-Modal Raw + Spectral Fusion

**Model:** `fusion_c`

Original conceptual description:

- raw EEG branch
- 2D scalogram branch
- gated fusion between temporal and spectral features

Important code clarification:

- the conceptual label originally referenced Swin
- the current implementation uses **EfficientNet-B0** on the 2D branch for speed
- the 1D branch uses **SNN-1D-Attention**

So the actual current design is:

- 1D raw EEG via SNN-1D-Attention
- 2D scalogram via EfficientNet-B0
- gated multimodal fusion head

Status in W&B:

- **currently running**
- not yet available as a finalized comparison number in the CSV

Why it matters:

- this is the most direct test of the project’s central multimodal hypothesis

Architecture summary:

- Input A: raw EEG sequence
- Input B: scalogram image of the same epoch
- Raw branch: `snn_1d_attn` extracts temporal features directly from the waveform
- 2D branch: EfficientNet-B0 extracts compact spectral features from the scalogram
- Projection: both branches are projected into a common embedding dimension
- Fusion: a gated multimodal fusion module combines the two embeddings
- Head: fused representation is classified into the 5 sleep stages

Why it matters scientifically:

- this is the cleanest architecture for asking whether temporal and spectral EEG views are complementary enough to beat single-branch models

### 9.5 `snn_fusion_early`

**Model:** `snn_fusion_early`

Concept:

- features from the 1D and 2D branches are fused before final classification
- strongest feature mixing among the SNN fusion family

Status:

- present in W&B
- marked finished
- but the CSV row does not contain a clean final metric summary

Interpretation:

- it should be mentioned as **tried**
- but not treated as a benchmark result on the same level as `fusion_a` or `fusion_b`

### 9.6 `snn_fusion_late`

**Model:** `snn_fusion_late`

Concept:

- each branch predicts separately
- predictions are combined later as an ensemble

Status:

- present in W&B
- **crashed**

Interpretation:

- this is a tried architecture idea, but not a completed benchmark result

### 9.7 `snn_fusion_gated`

**Model:** `snn_fusion_gated`

Concept:

- adaptive routing between the two modalities
- tries to learn when each branch should dominate

Status:

- implemented in the repo
- **no clean W&B run appears in the latest CSV**

Interpretation:

- include it in the report as an implemented branch
- do not present it as a completed empirical result

### 9.8 `quantum_snn_fusion_early`

Concept:

- combine raw SNN features and quantum-processed 2D features through early fusion

Status:

- implemented in the repo
- **no clean W&B run in the latest CSV**

Interpretation:

- active architecture direction
- not yet a benchmark result

### 9.9 `quantum_snn_fusion_gated`

Concept:

- adaptive multimodal fusion between raw SNN and quantum 2D features

Status:

- implemented in the repo
- **no clean W&B run in the latest CSV**

Interpretation:

- one of the more novel research directions in the codebase
- should be reported as present and important, but not yet completed empirically

### 9.10 Architecture Notes on Other Important Models

The following single-branch models are also important to understand because they anchor the benchmark.

#### `convnext`

Architecture summary:

- Input: 2D EEG scalogram
- Backbone: ConvNeXt-Tiny
- Processing style: modernized convolutional hierarchy with patch-style stem, depthwise blocks, and stage-wise feature abstraction
- Head: pooled feature vector followed by classifier

Why it matters:

- it is one of the strongest pure classical baselines
- it shows that a well-tuned CNN can remain highly competitive even against transformers and fusion models

#### `snn_1d_attn`

Architecture summary:

- Input: raw EEG signal
- Backbone: 1D spiking network
- Temporal modeling: spike-based temporal feature extraction over the raw sequence
- Attention: an attention block helps the model focus on the most informative temporal segments or channels
- Head: final feature vector is classified into sleep stages

Why it matters:

- it is the strongest finished raw 1D model in the latest rerun set
- it demonstrates that raw EEG can work well without forcing everything through a 2D transform

#### `quantum_ring_RXY` and `quantum_full_RXY`

Architecture summary:

- Input: 2D EEG scalogram
- Front-end: classical 2D feature extractor
- Bottleneck: quantum-inspired circuit layer with `RXY` rotations
- Entanglement:
  - `ring` connects qubits locally in a loop
  - `full` uses denser all-to-all style entanglement
- Head: quantum-transformed features are sent to the final classifier

Why they matter:

- these are the focused quantum candidates selected from the larger 14-model sweep
- they represent the main quantum benchmark line in the later paper-style runs

#### `tcanet`

Architecture summary:

- Input: raw EEG signal
- Core idea: temporal convolution and attention-style processing specialized for EEG sequence modeling
- Current repo direction: integrated as a clean raw-signal architecture and extended with a real quantum bottleneck path
- Head: learned temporal representation goes to the final sleep-stage classifier

Why it matters:

- even though current BOAS results are not yet strong, it remains a strategically important architecture because it is a more EEG-specialized temporal model than the generic baseline families

---

## 10. Status Summary: Completed vs In-Progress vs Implemented

### 10.1 Completed and Tracked with Usable Metrics

- `snn_lif_resnet`
- `snn_qif_resnet`
- `snn_lif_vit`
- `snn_qif_vit`
- `swin`
- `vit`
- `deit`
- `efficientnet`
- `convnext`
- all 14 2D quantum variants
- `snn_1d_lif`
- `snn_1d_attn`
- `spiking_vit_1d`
- all 14 1D quantum variants
- `fusion_a`
- `fusion_b`
- `tcanet`

### 10.2 In Progress or Incomplete

- `fusion_c`:
  - active running experiment in latest CSV
- `snn_fusion_early`:
  - finished state but without clean usable final metric summary in the export
- `snn_fusion_late`:
  - crashed

### 10.3 Implemented in the Repo but Not Represented by a Clean W&B Benchmark in the Latest CSV

- `snn_fusion_gated`
- `quantum_snn_fusion_early`
- `quantum_snn_fusion_gated`

This distinction is important for mentor discussion because it prevents overstating what is already empirically finished.

---

## 11. Main Findings to Present

### 11.1 2D Scalograms Are the Strongest Overall Foundation

Across the full project history, the best historical values are mostly from:

- strong 2D classical backbones
- 2D quantum variants
- 2D fusion models

### 11.2 Raw 1D EEG Became Much More Credible in the Newer Runs

The jump in 1D SNN performance is one of the strongest findings in the project:

- raw 1D is no longer just a weak exploratory branch
- it is now a serious contributor
- this makes multimodal fusion scientifically meaningful

### 11.3 Fusion Is the Most Promising Endgame

Fusion matters because:

- 2D is strong
- 1D improved sharply
- the modalities are complementary

That makes the fusion family the cleanest route to a final “best system” story.

### 11.4 Best Finished Latest Rerun

As of the latest CSV:

- **best finished latest rerun = `fusion_b`**
- accuracy = **82.44%**
- macro-F1 = **0.679**

### 11.5 Strong Historical Models Still Need Standardized Reruns

The project still needs a final standardized comparison among:

- top 2D classical models
- top 2D quantum models
- top 1D SNN models
- top fusion models

This is necessary because historical bests and later focused reruns were not all produced under one perfectly unified setup.

---

## 12. Recommended Next Steps

The most useful next steps are:

1. Finalize `fusion_c` and obtain a stable completed metric table.
2. Run clean benchmarked versions of:
   - `fusion_b`
   - `fusion_a`
   - `convnext`
   - `snn_1d_attn`
   - `snn_1d_lif`
   - top quantum 2D variants
3. Decide whether the weaker branches should stay in the final paper table:
   - `spiking_vit_1d`
   - QIF variants
   - TCANet in its current form
4. Add clean benchmark runs for the implemented-but-not-yet-reported fusion branches:
   - `snn_fusion_gated`
   - `quantum_snn_fusion_early`
   - `quantum_snn_fusion_gated`
5. Extend the validated pipeline beyond BOAS to Sleep-EDF for generalization testing.

---

## 13. Final Summary

This project has grown from an exploratory benchmark into a fairly rich EEG sleep staging research platform.

The major milestones so far are:

- a strong 2D scalogram benchmark foundation
- a complete 2D quantum sweep
- a complete 1D quantum sweep
- major improvement in raw 1D SNN performance
- several fusion strategies, with `fusion_b` currently the best finished latest rerun
- `fusion_c` as the most important active multimodal experiment
- additional fusion architectures already implemented and ready for fuller benchmarking

If summarized in one sentence:

> The project shows that 2D spectral modeling has been the strongest overall base, raw 1D modeling has improved enough to become genuinely useful, and multimodal fusion is now the most promising route to the final best-performing sleep staging system.
