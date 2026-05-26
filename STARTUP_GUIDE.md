# QSpikeXAI-Net — Startup and Execution Guide
> **Target Environment:** Ubuntu Linux · NVIDIA RTX 5050 GPU (or any CUDA-enabled GPU)  
> **Skill Level:** Beginners to Advanced (Zero-knowledge friendly)

This guide provides step-by-step instructions to clone, set up, and run the unified **QSpikeXAI-Net** EEG classification pipeline.

---

## 📋 1. Prerequisites (Target Environment)

Before running the codebase, make sure your Ubuntu environment has the following installed:

1. **NVIDIA Drivers & CUDA:**
   Since you are running on an **RTX 5050 GPU** (Ada Lovelace architecture), you need NVIDIA drivers supporting **CUDA 11.8** or **CUDA 12.1+** (CUDA 12.1+ is highly recommended).
   * Check your driver status by running:
     ```bash
     nvidia-smi
     ```
2. **System Dependencies:**
   Install Python3, Python virtual environment package, and Git:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-pip python3-venv git curl wget
   ```

---

## 🚀 2. Quick Start: Clone & Setup

Your friend can set up the entire project (including folder creation, Python dependencies, automated dataset downloads, and validation tests) using these simple steps.

### Step 2.1: Clone the Repository
```bash
git clone <your-github-repo-url>
cd EEG
```

### Step 2.2: Run the Setup Script
Run the unified setup script. This script will automatically create a Python virtual environment (`venv`), install dependencies, prompt for W&B login, download public datasets, and run model shape tests:
```bash
bash setup_and_run.sh
```

---

## 📊 3. Dataset Matrix & Directories

The codebase is organized to search for datasets in the `data/` directory inside the repository. Below is a comprehensive overview of the four tasks:

### Dataset Overview Table

| Task ID | Clinical Disorder | Dataset Name | Download Method | Format | Direct Web Link / Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`sleep_apnea`** | Obstructive Sleep Apnea | PhysioNet Apnea-ECG | **Programmatic** (via copy/setup) | `.dat`, `.hea`, `.apn` | [PhysioNet Apnea-ECG](https://physionet.org/content/apnea-ecg/1.0.0/) |
| **`schizophrenia`** | Schizophrenia | PhysioNet EEG-Schizophrenia | **Programmatic** (automatic) | `.edf` | [PhysioNet EEG-SCZ](https://physionet.org/content/eeg-schizophrenia/1.0.0/) |
| **`mci`** | Mild Cognitive Impairment | CAUEEG Dataset | **Programmatic** (clone + script) | `.edf` (BIDS) | [GitHub caueeg-dataset](https://github.com/ipis-mjkim/caueeg-dataset) |
| **`depression`** | Major Depressive Disorder | MODMA Dataset | ⚠️ **Manual** (Registration required) | `.mat` | [MODMA Portal](http://modma.lzu.edu.cn/data/index/) |
| **`depression`** (Alt) | Major Depressive Disorder | Mumtaz Figshare | **Manual** (Direct zip download) | `.mat` | [Figshare EEG Depression](https://figshare.com/articles/dataset/eeg_data/4244883) |

---

## 📁 4. Where to Place the Datasets (Directory Structure)

Your friend should place the dataset files exactly as shown below:

```
EEG/ (Root repository folder)
├── setup_and_run.sh
├── requirements.txt
├── STARTUP_GUIDE.md
├── data/
│   ├── apnea-ecg/
│   │   ├── a01.dat
│   │   ├── a01.hea
│   │   ├── a01.apn
│   │   └── (Rest of Apnea-ECG files...)
│   │
│   ├── eeg-schizophrenia/
│   │   ├── s01.edf
│   │   ├── h01.edf
│   │   └── (Rest of Schizophrenia files...)
│   │
│   ├── caueeg/
│   │   ├── download_caueeg.py
│   │   ├── (Run "python download_caueeg.py" inside here to download EDFs)
│   │   └── (Move the downloaded .edf files directly into data/caueeg/)
│   │
│   └── depression/
│       ├── (Place MODMA MAT files, e.g., MODMA_EEG_1.mat, here)
│       └── (OR extract Mumtaz .mat files directly into this directory)
```

> [!TIP]
> **Disk Space Optimization:** If your friend's main partition has low space, they can store datasets on another hard drive and link it:
> `ln -s /path/to/large/drive/datasets data`

---

## 🧪 5. Verification: Pipeline Mock Dry-Run

Before waiting for gigabytes of datasets to download, your friend can run a **Mock Run** to verify that their RTX 5050 GPU, SNN, and Quantum modules are working correctly. 

Our data loader falls back to **safe mock data generation** when real data folders are missing or empty.

### Run Model Shape & Compilation Tests:
Verify that PyTorch, SNN, and Quantum circuits initialize and output correct shapes:
```bash
# Activate environment
source venv/bin/activate

# Run tests
python3 -m unittest tests/test_models.py
python3 -m unittest tests/test_dataloaders.py
python3 -m unittest tests/test_xai.py
```

### Run Mock Proposed Model Training:
Train QSpikeXAI-Net on synthetic/mock Sleep Apnea data for 2 epochs:
```bash
python3 experiments/run_proposed.py --task sleep_apnea --data-dir data/apnea-ecg --epochs 2 --folds 1
```

---

## 📈 6. Production Training & Execution Commands

Once the datasets are configured, use the following commands to execute training.

Ensure your virtual environment is active first:
```bash
source venv/bin/activate
```

### 1. Proposed Model (QSpikeXAI-Net)
Runs the hybrid SNN + Quantum VQC model:
```bash
python3 experiments/run_proposed.py --task sleep_apnea --data-dir data/apnea-ecg/ --epochs 80 --folds 5 --use-wandb
```
*(Replace `sleep_apnea` with `schizophrenia`, `mci`, or `depression` as needed).*

### 2. Baseline Comparisons
Train comparative baseline architectures:
```bash
# Train EEGNet
python3 experiments/run_baselines.py --task sleep_apnea --model eegnet --data-dir data/apnea-ecg/ --use-wandb

# Train EEGTCNet
python3 experiments/run_baselines.py --task sleep_apnea --model eeg_tcnet --data-dir data/apnea-ecg/ --use-wandb

# Train ResNet1D
python3 experiments/run_baselines.py --task sleep_apnea --model resnet1d --data-dir data/apnea-ecg/ --use-wandb

# Train ViT1D
python3 experiments/run_baselines.py --task sleep_apnea --model vit1d --data-dir data/apnea-ecg/ --use-wandb
```

### 3. Ablation Runs
Test sub-components individually (SNN-only, Quantum-only, or Concatenated-fusion):
```bash
# SNN-only
python3 experiments/run_ablations.py --task sleep_apnea --model snn_only --data-dir data/apnea-ecg/ --use-wandb

# Quantum-only
python3 experiments/run_ablations.py --task sleep_apnea --model quantum_only --data-dir data/apnea-ecg/ --use-wandb
```

### 4. Explainability (XAI) Attribution Extraction
Run the XAI module using a trained model checkpoint to generate channel/temporal saliency and VQC gate attributions:
```bash
python3 experiments/run_xai.py --task sleep_apnea --checkpoint results/best_model_sleep_apnea_fold0.pt --data-dir data/apnea-ecg/
```

---

## 📈 7. Weights & Biases (W&B) Logging

When you append `--use-wandb` to any training command, the training script logs metrics to W&B.
* During the first execution of `setup_and_run.sh`, the terminal will ask if you want to initialize W&B.
* If you chose to skip it during setup, you can log in anytime by running:
  ```bash
  python3 -m wandb login
  ```
  Paste your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize) when prompted.
