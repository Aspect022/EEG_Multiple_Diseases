# EEG Project Structure and Current Results

Last reviewed: 2026-05-26  
Workspace: `D:\Projects\AI-Projects\EEG`

## 1. What this project is

This repository is a research codebase for EEG and sleep-disorder classification. It currently contains several generations of pipeline work:

- Sleep staging on EEG, mainly 5-class classification: `Wake`, `N1`, `N2`, `N3`, `REM`.
- Sleep apnea classification, mostly planned as 4-class severity: `Healthy`, `Mild`, `Moderate`, `Severe`.
- Model families spanning spiking neural networks, transformer baselines, quantum-classical models, and fusion models.
- Saved experiment outputs from local JSON runs and exported W&B CSV runs.

The codebase is not just one clean application. It is a research workspace with:

- a current unified pipeline idea in `unified_pipeline.py`;
- a larger, more complete sleep-staging benchmark pipeline in `pipeline.py`;
- task-specific apnea and Sleep-EDF pipelines;
- modular model/data/training code under `src/` and `sleep_apnea/`;
- many historical docs and result exports.

## 2. Current top-level structure

```text
EEG/
|-- README.md
|-- requirements.txt
|-- pipeline.py
|-- unified_pipeline.py
|-- sleep_apnea_pipeline.py
|-- sleep_edf_pipeline.py
|-- paper_pipeline.py
|-- comprehensive_snn_pipeline.py
|-- run*.sh
|-- allResults.csv
|-- data/
|-- docs/
|-- outputs/
|-- results/
|-- Scripts/
|-- sleep_apnea/
|-- src/
|-- tests/
|-- notebooks/
|-- venv/
```

### Important top-level files

| File | Role |
|---|---|
| `pipeline.py` | Largest sleep-staging experiment runner. Defines many BOAS experiments: SNN, SNN-1D, quantum, transformers, fusion, and conditional-routing models. Outputs to `outputs/results` by default. |
| `unified_pipeline.py` | Newer single-interface pipeline concept for both `sleep_staging` and `sleep_apnea`. It currently has a smaller in-file model registry and shared trainer. |
| `sleep_apnea_pipeline.py` | Standalone sleep-apnea severity pipeline with CNN, ResNet18 transfer learning, and ViT+BiLSTM classes. |
| `sleep_edf_pipeline.py` | Sleep-EDF experiment pipeline with model validation and multimodal loader support. |
| `paper_pipeline.py` | Preset-oriented runner for paper-style experiments. |
| `comprehensive_snn_pipeline.py` | SNN-focused staged runner for broader SNN experiments. |
| `run*.sh` | Convenience launch scripts for different experiment families. |
| `allResults.csv` | Latest broad W&B-style result export at repo root. This is important for understanding the strongest reported runs. |

## 3. Main source package: `src/`

`src/` is the main reusable code package for sleep-staging and general EEG experiments.

```text
src/
|-- data/
|-- evaluation/
|-- models/
|-- training/
|-- utils/
```

### `src/data/`

This folder contains dataset loaders and signal transformations.

| File | Purpose |
|---|---|
| `boas_dataset.py` | BOAS/OpenNeuro `ds005555` loader. Uses BIDS-style subject folders, 30-second epochs, 6 EEG channels, and subject-level train/val/test splitting. |
| `sleep_edf_dataset.py` | Sleep-EDF Expanded utilities. Finds PSG/Hypnogram pairs, extracts 30-second epochs, pads/trims channels to fixed shape, and supports raw/scalogram/multimodal loaders. |
| `apnea_dataset.py` | Apnea dataset support in the main package. |
| `ptbxl_dataset.py` | PTB-XL-related dataset support. |
| `dataset.py` | More general dataset utilities. |
| `preprocessing.py`, `preprocessing_grayscale.py`, `transforms.py` | Signal preprocessing and scalogram/spectrogram transformation code. |

The sleep-staging path expects 30-second windows. BOAS uses target EEG channels such as `F3`, `F4`, `C3`, `C4`, `O1`, `O2`; Sleep-EDF normalizes records into a fixed 6-channel shape.

### `src/models/`

This is the main model library.

```text
src/models/
|-- baseline/
|-- fusion/
|-- quantum/
|-- snn/
|-- snn_1d/
|-- transformer/
|-- tcanet_clean.py
```

Model groups:

- `baseline/`: conventional CNN baseline.
- `snn/`: 2D spiking models such as spiking ResNet and spiking ViT.
- `snn_1d/`: raw-signal spiking models, LIF neurons, attention, and 1D spiking ViT.
- `quantum/`: hybrid classical/quantum circuit models, including 1D and 2D variants.
- `transformer/`: Swin, ViT, DeiT, EfficientNet, ConvNeXt wrappers.
- `fusion/`: fusion models combining backbones or modalities:
  - `fusion_a`: Swin + ConvNeXt.
  - `fusion_b`: hybrid multi-backbone fusion including quantum features.
  - `fusion_c`: multimodal raw EEG + scalogram fusion.
  - `gated_fusion.py`: adaptive/gated fusion logic.
  - `conditional_routing_fusion.py`: fast/slow routing idea.
- `tcanet_clean.py`: TCANet-style temporal model.

### `src/training/`

Training utilities live here.

| File | Purpose |
|---|---|
| `trainer.py` | General training loop utilities. |
| `research_trainer.py` | Research-grade training support: cross-validation, checkpoints, mixed precision, W&B/TensorBoard hooks, early stopping, and metrics logging. |
| `multimodal_trainer.py` | Training support for models that consume multiple inputs, such as raw EEG plus scalogram. |

### `src/evaluation/`

`metrics.py` computes a large metric set: accuracy, balanced accuracy, macro/weighted precision and recall, F1, confusion-derived one-vs-rest metrics, MCC, Cohen kappa, hamming loss, log loss, entropy, and AUC-ROC.

## 4. Sleep-apnea package: `sleep_apnea/`

`sleep_apnea/` is a modular implementation for apnea-specific work.

```text
sleep_apnea/
|-- configs/default_config.yaml
|-- data/
|-- models/
|-- training/
|-- utils/
|-- run_apnea.py
|-- README.md
|-- QUICK_START.md
```

Important pieces:

- `data/apnea_ecg_dataset.py`: PhysioNet Apnea-ECG support.
- `data/shhs_dataset.py`: SHHS support.
- `models/custom_cnn.py`: apnea CNN baseline.
- `models/resnet18_transfer.py`: ResNet18 transfer learning model.
- `models/vit_bilstm.py`: ViT plus BiLSTM hybrid, treated in docs as the main apnea contribution.
- `models/ssl_pretrainer.py`: self-supervised pretraining components.
- `training/apnea_trainer.py`: apnea training loop.
- `training/ssl_trainer.py`: SSL training loop.
- `utils/ahi_computation.py`, `utils/severity_labels.py`: AHI and severity-label helpers.

## 5. Data currently present

The local `data/` folder contains:

```text
data/
|-- apnea-ecg-database-1.0.0/
|-- sleep-edf/
```

Observed datasets:

- `data/apnea-ecg-database-1.0.0`: PhysioNet Apnea-ECG files, including `.dat`, `.hea`, `.apn`, `.qrs`, `.xws`, challenge files, and record lists.
- `data/sleep-edf`: at least one Sleep-EDF PSG/Hypnogram pair: `SC4001E0-PSG.edf` and `SC4001EC-Hypnogram.edf`.

The BOAS dataset is referenced heavily by `pipeline.py` and docs as OpenNeuro `ds005555`, but it was not visible as a populated local `data/ds005555` directory during this review.

## 6. Pipeline entrypoints

### `pipeline.py`: broad BOAS sleep-staging benchmark

This is the most complete experiment orchestrator for the sleep-staging benchmark.

Default behavior:

```bash
python pipeline.py --epochs 30 --batch-size 128
```

Useful quick test:

```bash
python pipeline.py --epochs 1 --models snn_lif_resnet --max-subjects 5 --skip-download
```

Key behavior:

- Downloads/verifies BOAS unless `--skip-download` is used.
- Defines about 40+ experiment configurations.
- Supports model selection via `--models`.
- Retries with smaller batch sizes after CUDA OOM.
- Saves per-model result folders under `outputs/results`.

Main experiment groups in `pipeline.py`:

- 2D SNN: `snn_lif_resnet`, `snn_qif_resnet`, `snn_lif_vit`, `snn_qif_vit`.
- Transformers: `swin`, `vit`, `deit`, `efficientnet`, `convnext`.
- 2D quantum: `quantum_{ring|full}_{RX|RY|RZ|RXY|RXZ|RYZ|RXYZ}`.
- 1D SNN: `snn_1d_lif`, `snn_1d_attn`, `spiking_vit_1d`.
- 1D quantum: `quantum_1d_{ring|full}_{...}`.
- Fusion: `fusion_a`, `fusion_b`, `fusion_c`, `snn_fusion_*`, `quantum_snn_fusion_*`.
- Conditional routing: `conditional_routing`.

### `unified_pipeline.py`: newer single CLI idea

This file tries to make one interface for both sleep staging and sleep apnea.

Example usage from docs:

```bash
python unified_pipeline.py --task sleep_staging --model snn_lif_resnet --dataset boas
python unified_pipeline.py --task sleep_apnea --model vit_bilstm --dataset shhs --ssl-pretrain
```

Current model registry in the file:

- Sleep staging: `snn_lif_resnet`, `snn_vit`.
- Sleep apnea: `cnn`, `resnet18`, `vit_bilstm`.

Important note: this file is conceptually cleaner, but the full mature model zoo is still mostly wired through `pipeline.py` and `src/models/`.

### `sleep_apnea_pipeline.py`: standalone apnea runner

Example:

```bash
python sleep_apnea_pipeline.py --model cnn --data-dir data/shhs --epochs 30
python sleep_apnea_pipeline.py --model vit_bilstm --data-dir data/shhs --ssl-pretrain --epochs 30
```

Supported models:

- `cnn`
- `resnet18`
- `vit_bilstm`

The implementation includes dataset classes, transforms, model classes, and `ApneaTrainer` in one large script. A more modular version also exists under `sleep_apnea/`.

### `sleep_edf_pipeline.py`: Sleep-EDF path

This pipeline supports Sleep-EDF data validation, model forward validation, raw/scalogram/multimodal dataloaders, and experiment execution. It is useful for cross-dataset staging validation.

## 7. Results and artifacts

There are three main result locations.

### `outputs/results/`

This contains local JSON/CSV artifacts from earlier runs. Each model folder has:

```text
outputs/results/<experiment>/
|-- results.json
|-- error.json              # present for failed or partially failed runs
|-- fold_0/training_history.csv
```

Top local `outputs/results` runs by `results.json` accuracy:

| Rank | Experiment | Accuracy | Macro F1 | AUC-ROC | Best epoch |
|---:|---|---:|---:|---:|---:|
| 1 | `swin` | 0.8462 | 0.6958 | 0.9591 | 11 |
| 2 | `efficientnet` | 0.8402 | 0.6836 | 0.9513 | 10 |
| 3 | `quantum_ring_RXY` | 0.8398 | 0.6461 | 0.9512 | 29 |
| 4 | `quantum_ring_RXZ` | 0.8371 | 0.6594 | 0.9519 | 29 |
| 5 | `quantum_ring_RXYZ` | 0.8365 | 0.6557 | 0.9511 | 26 |
| 6 | `quantum_full_RXZ` | 0.8355 | 0.6612 | 0.9516 | 22 |
| 7 | `quantum_full_RXYZ` | 0.8353 | 0.6331 | 0.9484 | 30 |
| 8 | `quantum_full_RXY` | 0.8343 | 0.6566 | 0.9490 | 19 |

Interpretation:

- These are mostly March 16 local outputs.
- Transformer and 2D quantum models are the strongest in this saved local set.
- Some folders contain `error.json`, so a result folder can represent partial success plus an error.

### Root `allResults.csv`

This appears to be a later W&B-style export and contains stronger later results from March 22-23.

Top rows by `best_accuracy`:

| Rank | Run | Best accuracy | Final macro F1 | Final AUC | Best epoch |
|---:|---|---:|---:|---:|---:|
| 1 | `snn_1d_attn_fold0` | 0.9474 | 0.8388 | 0.9888 | 26 |
| 2 | `quantum_1d_ring_RYZ_fold0` | 0.9456 | 0.8399 | 0.9667 | 30 |
| 3 | `snn_1d_lif_fold0` | 0.9374 | 0.8185 | 0.9892 | 28 |
| 4 | `quantum_1d_full_RXY_fold0` | 0.9017 | 0.7673 | 0.9654 | 26 |
| 5 | `swin_fold0` | 0.8462 | 0.6958 | 0.9591 | 11 |
| 6 | `tcanet_fold0` | 0.8441 | 0.6658 | 0.9588 | 30 |
| 7 | `efficientnet_fold0` | 0.8402 | 0.6836 | 0.9513 | 10 |
| 8 | `swin_fold0` | 0.8400 | 0.6982 | 0.9610 | 9 |

Interpretation:

- This export suggests later 1D SNN and 1D quantum experiments became the best reported runs.
- The metrics are exported from W&B and should be treated as the strongest available summary, but they do not all have matching local `outputs/results/<model>/results.json` folders.
- There are duplicate run names such as `swin_fold0`, likely from reruns.

### `results/`

This folder stores W&B export CSVs:

- `results/ALLResults.csv`
- `results/wandb_export_2026-03-22T13_09_19.344+05_30.csv`
- `results/wandb_export_2026-03-23T00_46_06.065+05_30.csv`

The March 23 export top ranking is closer to the older local outputs than root `allResults.csv`; the root CSV includes later/high-performing 1D runs.

## 8. What the current results mean

There are two result narratives in the repository:

1. Older docs and `outputs/results` emphasize 2D transformer/quantum/fusion models, with best local JSON accuracy around 84.6% for `swin`.
2. The root `allResults.csv` shows later W&B-exported runs where 1D SNN and 1D quantum models report 90-95% best accuracy.

Because of this, the current state should be described carefully:

- Best local JSON artifact: `swin`, about 84.6% accuracy.
- Best root W&B export row: `snn_1d_attn_fold0`, about 94.7% best accuracy.
- Best W&B exported macro F1 among the top rows: `quantum_1d_ring_RYZ_fold0`, about 0.8399 macro F1.
- Strong AUC appears in `snn_1d_lif_fold0` and `snn_1d_attn_fold0`, around 0.989.

Before writing a paper table, the result sources should be reconciled into one canonical experiment ledger with dataset, split, command, seed, fold count, and artifact path for each run.

## 8.1 Local failure artifacts

Some experiment folders in `outputs/results/` contain `error.json` even when a `results.json` is also present. Treat those folders as partial/fragile until rerun or verified.

Observed local errors:

| Experiments | Error pattern | Meaning |
|---|---|---|
| `quantum_ring_RX`, `quantum_ring_RXY`, `quantum_ring_RXYZ`, `quantum_ring_RXZ`, `quantum_ring_RY`, `quantum_ring_RYZ`, `quantum_ring_RZ`, `snn_lif_vit`, `swin` | `num_samples should be a positive integer value, but got num_samples=0` | A dataloader/sampler split was empty during at least one attempted run. This usually points to incomplete dataset split, cache issue, or rerun state mismatch. |
| `snn_lif_resnet` | `'precision'` | A metric-key mismatch occurred while logging or summarizing results. |
| `snn_qif_resnet`, `snn_qif_vit` | CUDA out of memory | Long multi-model run exhausted GPU memory despite a large A100-class GPU. Batch retry/cleanup may not have been enough for these specific attempts. |

This does not necessarily invalidate the matching `results.json`; it means the folder records more than one attempt or a partially failed pipeline stage.

## 8.2 How a current sleep-staging run flows

The most complete active flow is:

```text
pipeline.py
  |
  |-- parse CLI args
  |-- verify/download BOAS dataset
  |-- select experiment configs from EXPERIMENT_DEFS
  |
  |-- for each experiment:
      |
      |-- run_experiment()
          |
          |-- choose data mode
          |     |-- 1d   -> create_boas_dataloaders(..., transform=None)
          |     |-- 2d   -> cached scalograms if available, otherwise CWT transform
          |     |-- both -> create_boas_multimodal_dataloaders()
          |
          |-- create_model(exp_config)
          |-- compute class weights from train labels
          |-- choose SNN or non-SNN hyperparameters
          |-- create ResearchConfig
          |-- train with FoldTrainer or MultiModalFoldTrainer
          |-- write results.json or error.json
  |
  |-- generate experiment_summary.json and experiment_summary.md
```

The main reusable training classes are:

- `src.training.research_trainer.FoldTrainer` for single-input models.
- `src.training.multimodal_trainer.MultiModalFoldTrainer` for paired raw/scalogram models.
- `src.evaluation.metrics.compute_all_metrics` for final metrics.

Important implementation detail: `pipeline.py` runs one fold for speed, even though `ResearchConfig` supports cross-validation-style settings.

## 8.3 How a Sleep-EDF run flows

`sleep_edf_pipeline.py` follows the same broad training architecture as `pipeline.py`, but swaps in Sleep-EDF dataset loading:

```text
sleep_edf_pipeline.py
  |
  |-- verify_sleep_edf_dataset()
  |-- resolve model selector into experiment configs
  |-- optionally validate model forward passes
  |-- build_sleep_edf_loaders()
  |-- create_model() using the shared pipeline model factory
  |-- train with FoldTrainer or MultiModalFoldTrainer
  |-- write per-model results
```

Important operational note: the default validation requires a recommended minimum number of matched Sleep-EDF PSG/Hypnogram pairs. The local workspace currently shows only one pair, so a full Sleep-EDF training run would likely stop early unless more records are downloaded or the minimum is overridden in code.

## 8.4 Unified pipeline status

`unified_pipeline.py` is a useful design target, but it is not yet the true canonical runner.

Current state:

- It defines common task names, class names, a small model registry, and a shared `UnifiedTrainer`.
- It supports CLI flags such as `--task`, `--model`, `--dataset`, `--ssl-pretrain`, and `--run-all`.
- Its `main()` currently prints `[INFO] Using placeholder data - implement <dataset> loader` and simulates only a short dummy training loop.

Practical implication:

- Use `pipeline.py` or `sleep_edf_pipeline.py` for real current benchmark runs.
- Use `unified_pipeline.py` as the file to evolve when consolidating the project into a cleaner long-term interface.

## 8.5 Model wiring map

This is the practical model-factory map from `pipeline.py`.

| Experiment `type` | Constructor path |
|---|---|
| `snn` | `src.models.snn.create_spiking_resnet` |
| `snn_vit` | `src.models.snn.create_spiking_vit` |
| `snn_1d` | `src.models.snn_1d.create_snn_1d_lif` or `create_snn_1d_attention` |
| `spiking_vit_1d` | `src.models.snn_1d.create_spiking_vit_1d` |
| `quantum` | `src.models.quantum.create_hybrid_quantum_cnn` |
| `quantum_1d` | `src.models.quantum.quantum_1d.create_quantum_1d` |
| `swin` | `src.models.transformer.create_swin_classifier` |
| `vit` | `src.models.transformer.create_vit_classifier` |
| `deit` | `src.models.transformer.create_deit_classifier` |
| `efficientnet` | `src.models.transformer.create_efficientnet_classifier` |
| `convnext` | `src.models.transformer.create_convnext_classifier` |
| `fusion_a` | `src.models.fusion.create_fusion_a` |
| `fusion_b` | `src.models.fusion.create_fusion_b` |
| `fusion_c` | `src.models.fusion.create_fusion_c` |
| `snn_fusion_early` | `src.models.fusion.create_early_fusion_complete` |
| `snn_fusion_late` | `src.models.fusion.create_late_fusion_complete` |
| `snn_fusion_gated` | `src.models.fusion.create_gated_fusion_complete` |
| `quantum_snn_fusion_early` | `src.models.fusion.create_quantum_snn_fusion_early` |
| `quantum_snn_fusion_gated` | `src.models.fusion.create_quantum_snn_fusion_gated` |
| `conditional_routing` | `src.models.fusion.conditional_routing_fusion.create_conditional_routing` |
| `tcanet` | Intended `TCANetClean`, currently imported in `pipeline.py` as `from tcanet_clean import TCANetClean`; the actual visible file is `src/models/tcanet_clean.py`, so this import path should be verified before running `tcanet` from `pipeline.py`. |

## 8.6 Fusion model details

The fusion family is where much of the novelty lives.

| Model | Inputs | Branches | Fusion idea |
|---|---|---|---|
| `fusion_a` | 2D scalogram | Swin-Tiny + ConvNeXt-Tiny | Gated fusion of two 768-d feature streams. |
| `fusion_b` | 2D scalogram | Swin-Tiny + ConvNeXt-Tiny + DeiT-Small + quantum features | Multi-stream fusion combining local attention, CNN hierarchy, global attention, and quantum measurements. |
| `fusion_c` | raw EEG + scalogram | SNN-1D + Swin | Multimodal fusion between temporal raw signal and 2D scalogram representation. |
| `snn_fusion_*` | raw EEG + scalogram | SNN-1D + spiking ResNet | Early, late, or gated fusion variants. |
| `quantum_snn_fusion_*` | raw EEG + scalogram | SNN-1D + 2D quantum model | SNN temporal features fused with quantum scalogram features. |
| `conditional_routing` | raw EEG | SNN-1D attention + quantum-1D | Confidence gate learns how much to rely on a fast SNN path versus slower quantum features. |

The conditional-routing model is especially useful for future deployment work because it creates a path toward adaptive inference cost.

## 9. Documentation currently present

`docs/` is extensive and includes:

- `UNIFIED_PIPELINE_README.md`: unified pipeline concept and CLI.
- `EXPERIMENT_STATUS_OVERVIEW.md`: March 2026 status, planned experiments, publication ideas.
- `LATEST_RESULTS_ANALYSIS.md`: analysis around March 22 results.
- `COMPLETE_MODEL_DOCUMENTATION.md`: model-by-model documentation and older results.
- `PIPELINE_COMPARISON.md`: old multi-pipeline versus unified-pipeline direction.
- `SLEEP_APNEA_*.md`: apnea plan, quick start, implementation summary.
- `SNN_*`, `ROOT_CAUSE_ANALYSIS.md`, `OPTIMIZATION_CHANGES.md`: debugging and optimization history.
- `reports/`: mentor progress report and detailed model analysis in Markdown/HTML.

Some docs contain older numbers that conflict with later CSV exports. They are useful for history, but the result CSVs and local JSON outputs should be considered closer to source-of-truth artifacts.

## 10. Tests

`tests/` currently contains smoke-style scripts:

- `test_imports.py`: imports required packages and prints environment status.
- `test_snn_output.py`: manually checks spiking ResNet output on CUDA with random input. It uses an absolute server path and is not a standard portable pytest test.
- `test_sleep_edf.py`: validates Sleep-EDF dataset loading, multimodal batches, and model forward checks.
- `smoke_test.py`: general smoke testing.

These tests are helpful, but the suite is not yet a clean automated CI suite. Some tests assume CUDA, local data, or server-specific paths.

## 11. Known cleanup issues

These are not blockers, but they matter before expanding the project.

1. `pipeline.py` is the richest pipeline, while `unified_pipeline.py` is the cleanest interface. They are not fully merged yet.
2. Apnea code exists both as a large script (`sleep_apnea_pipeline.py`) and as a package (`sleep_apnea/`).
3. Result sources are split across `outputs/results`, `results/*.csv`, `outputs/WandDB/*.csv`, and `allResults.csv`.
4. Some docs contain stale expected results or older completed results.
5. Several generated/cache folders are present locally: `__pycache__`, `pytest-cache-files-*`, and `venv`.
6. Some text files show mojibake for emoji/symbols, likely from encoding mismatch.
7. The local BOAS data directory referenced by the main sleep-staging pipeline was not visible in `data/`; only Apnea-ECG and Sleep-EDF were visible.

## 12. Best expansion path from here

For future work, the cleanest path is:

1. Choose a canonical pipeline direction.
   - If speed matters now, extend `pipeline.py` because it already knows the full model zoo.
   - If maintainability matters most, move the complete `pipeline.py` registry and model creation into the `unified_pipeline.py` style.

2. Create a canonical experiment ledger.
   - One table or JSON file should track: run name, dataset, split, fold, seed, command, git commit, output path, accuracy, macro F1, balanced accuracy, AUC, and notes.

3. Normalize result folders.
   - Prefer one structure such as:

```text
outputs/
|-- experiments/
|   |-- sleep_staging/
|   |   |-- boas/
|   |   |   |-- <model>/<run_id>/
|   |-- sleep_apnea/
|       |-- apnea_ecg/
|       |-- shhs/
```

4. Make datasets explicit.
   - Add a small `DATASETS.md` or config file saying which datasets are present locally, where to download missing datasets, and which pipeline expects which layout.

5. Convert smoke tests into stable pytest tests.
   - Keep tiny CPU-only model-forward tests.
   - Mark data/CUDA tests with pytest markers.
   - Remove hardcoded server paths.

6. Reconcile apnea direction.
   - Either keep apnea as a separate package and call it from unified pipeline, or migrate its model/data/training modules into `src/`.

7. Update docs after result reconciliation.
   - The most important stale docs are result-analysis docs, because they may understate or contradict the later 1D SNN/quantum runs.

## 13. Quick mental model

Use this simple map when deciding what to change next:

```text
Research idea
  |
  |-- Dataset loader/change?
  |     -> src/data/ or sleep_apnea/data/
  |
  |-- New model?
  |     -> src/models/<family>/ or sleep_apnea/models/
  |
  |-- New training behavior?
  |     -> src/training/ or sleep_apnea/training/
  |
  |-- New benchmark run?
  |     -> pipeline.py now, unified_pipeline.py later if consolidating
  |
  |-- New result analysis?
        -> outputs/results + allResults.csv + results/*.csv
```

The project is in a strong research-prototype state: lots of architectures are implemented, many runs exist, and there is enough evidence to decide next experiments. The main thing to do before expanding heavily is to unify the pipeline and result-tracking story so new work does not add another parallel path.
