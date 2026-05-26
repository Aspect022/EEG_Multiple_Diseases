#!/bin/bash
# ============================================================================
# QSpikeXAI-Net — Startup, Setup, and Execution Script
# Target Platform: Ubuntu Linux · RTX 5050 GPU
# ============================================================================

set -e # Exit immediately on error

echo "========================================================================"
echo "          Starting Setup for QSpikeXAI-Net Pipeline"
echo "========================================================================"

# 1. Create Python Virtual Environment if it does not exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "Virtual environment 'venv' already exists."
fi

# 2. Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# 3. Install required packages
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 3.5. Weights & Biases Authentication
echo "------------------------------------------------------------------------"
echo "                 Weights & Biases (W&B) Setup"
echo "------------------------------------------------------------------------"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    python3 -c "import wandb; wandb.login(key='${WANDB_API_KEY}', relogin=True)"
    echo "[OK] W&B logged in via WANDB_API_KEY environment variable."
else
    if [ -t 0 ]; then
        echo "This project uses Weights & Biases (W&B) to track metrics and training curves."
        read -p "Would you like to log in to W&B now? [y/N]: " login_choice
        if [[ "$login_choice" =~ ^[Yy]$ ]]; then
            echo "Please retrieve your API key from https://wandb.ai/authorize"
            python3 -m wandb login
        else
            echo "W&B login skipped. Runs will log to CSV locally. To use W&B, run: python3 -m wandb login"
        fi
    else
        echo "Non-interactive shell detected. Skipping W&B authentication prompt."
        echo "To authenticate later, run: python3 -m wandb login"
    fi
fi

# 4. Create raw data directories
echo "Creating data directories..."
mkdir -p data/apnea-ecg
mkdir -p data/eeg-schizophrenia
mkdir -p data/caueeg
mkdir -p data/depression
mkdir -p results
mkdir -p results/xai

# 5. Dataset Setup
echo "------------------------------------------------------------------------"
echo "                       Setting Up Datasets"
echo "------------------------------------------------------------------------"

# Task 1: Sleep Apnea
if [ -d "data/apnea-ecg" ] && [ "$(ls -A data/apnea-ecg)" ]; then
    echo "[OK] Sleep Apnea ECG data is already present in 'data/apnea-ecg/'."
else
    echo "[!] Sleep Apnea ECG data missing. Copying from old repository folder if available..."
    if [ -d "../data/apnea-ecg-database-1.0.0" ]; then
        cp -r ../data/apnea-ecg-database-1.0.0/* data/apnea-ecg/
        echo "[OK] Copied Sleep Apnea ECG data."
    elif [ -d "data/apnea-ecg-database-1.0.0" ]; then
        cp -r data/apnea-ecg-database-1.0.0/* data/apnea-ecg/
        echo "[OK] Copied Sleep Apnea ECG data."
    else
        echo "[!] Apnea-ECG data could not be found locally. Please download from:"
        echo "    https://physionet.org/content/apnea-ecg/1.0.0/"
        echo "    And extract files into 'data/apnea-ecg/'."
    fi
fi

# Task 2: Schizophrenia (PhysioNet)
if [ "$(ls -A data/eeg-schizophrenia 2>/dev/null)" ]; then
    echo "[OK] Schizophrenia dataset is already present in 'data/eeg-schizophrenia/'."
else
    echo "Downloading Schizophrenia dataset from PhysioNet..."
    python3 -c "import wfdb; wfdb.dl_database('eeg-schizophrenia', 'data/eeg-schizophrenia/')"
    echo "[OK] Schizophrenia dataset downloaded."
fi

# Task 3: Mild Cognitive Impairment (CAUEEG)
if [ "$(ls -A data/caueeg 2>/dev/null)" ]; then
    echo "[OK] MCI dataset is already present in 'data/caueeg/'."
else
    echo "Cloning CAUEEG repository..."
    git clone https://github.com/ipis-mjkim/caueeg-dataset.git data/caueeg/
    echo "[!] To get the actual EDF files for CAUEEG, please run the download script inside CAUEEG:"
    echo "    cd data/caueeg && python download_caueeg.py"
    echo "    Then move the downloaded BIDS .edf files back into 'data/caueeg/' folder."
fi

# Task 4: Depression (MODMA / Mumtaz)
echo "[!] Depression Dataset requires manual download due to registration gates:"
echo "    1. Register at: http://modma.lzu.edu.cn/data/index/"
echo "    2. Download the resting-state EEG subset (.mat or .edf files)."
echo "    3. Place the files directly under 'data/depression/'."
echo "    Alternative: Download Mumtaz Figshare dataset from:"
echo "    https://figshare.com/articles/dataset/eeg_data/4244883"
echo "    Extract the files, and place .mat files under 'data/depression/'."

echo "------------------------------------------------------------------------"
echo "                     Running Model Structural Tests"
echo "------------------------------------------------------------------------"
# Run python tests to verify shapes and loaders
python3 -m unittest tests/test_models.py

echo "========================================================================"
echo "          Setup and Validation Complete! Ready to train."
echo "========================================================================"
echo "To run baseline comparisons, execute:"
echo "    python3 experiments/run_baselines.py --task sleep_apnea --model eegnet --data-dir data/apnea-ecg/"
echo ""
echo "To run proposed QSpikeXAI-Net model, execute:"
echo "    python3 experiments/run_proposed.py --task sleep_apnea --data-dir data/apnea-ecg/ --epochs 80 --folds 5"
echo ""
echo "To extract explainability maps (XAI) using a checkpoint, execute:"
echo "    python3 experiments/run_xai.py --task sleep_apnea --checkpoint results/best_model_sleep_apnea_fold0.pt --data-dir data/apnea-ecg/"
echo "========================================================================"
