#!/bin/bash
# ==========================================================================
# ECG Classification Pipeline — Automated Server Deployment Script
# ==========================================================================
#
# This script fully automates the research pipeline on a GPU server:
#   1. Downloads and verifies the PTB-XL dataset (with retry loop)
#   2. Creates a Python virtual environment
#   3. Installs all dependencies (including GPU-accelerated PennyLane)
#   4. Runs all 19 experiments sequentially
#   5. Commits and pushes results to GitHub
#
# Usage:
#   bash run.sh                     # Full run (30 epochs)
#   bash run.sh --epochs 5          # Quick test run
#   bash run.sh --models snn_lif_resnet,swin  # Specific models only
#
# ==========================================================================

set -euo pipefail

# ---- Configuration ----
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.001}"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/results}"
VENV_DIR="${VENV_DIR:-venv}"
PYTHON="${PYTHON:-python3}"
MODELS="${MODELS:-all}"

# Parse CLI args (override defaults)
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; shift ;;
    esac
done

echo "======================================================================"
echo "  ECG CLASSIFICATION PIPELINE — AUTOMATED SERVER RUN"
echo "======================================================================"
echo "  Epochs:     $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  LR:         $LR"
echo "  Data Dir:   $DATA_DIR"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Models:     $MODELS"
echo "  Started:    $(date)"
echo "======================================================================"


# ==========================================================================
# Step 1: Dataset Download + Verification Loop
# ==========================================================================

echo ""
echo "[STEP 1/5] Dataset Download & Verification"
echo "--------------------------------------------"

PTBXL_URL="https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
PTBXL_ZIP="$DATA_DIR/ptbxl.zip"

# Possible directory names after extraction
PTBXL_DIRS=(
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
    "ptb-xl-1.0.3"
    "ptb-xl"
)

verify_dataset() {
    for dir_name in "${PTBXL_DIRS[@]}"; do
        local check_path="$DATA_DIR/$dir_name"
        if [ -d "$check_path" ] && \
           [ -f "$check_path/ptbxl_database.csv" ] && \
           [ -f "$check_path/scp_statements.csv" ]; then
            local dat_count
            dat_count=$(find "$check_path" -name "*.dat" 2>/dev/null | wc -l)
            if [ "$dat_count" -ge 100 ]; then
                echo "  [OK] Dataset verified: $check_path ($dat_count signal files)"
                return 0
            fi
        fi
    done
    return 1
}

mkdir -p "$DATA_DIR"

MAX_RETRIES=3
attempt=1

while [ $attempt -le $MAX_RETRIES ]; do
    if verify_dataset; then
        break
    fi

    echo "  [DOWNLOAD] Attempt $attempt/$MAX_RETRIES..."

    # Download with resume support
    if command -v wget &> /dev/null; then
        wget -c -O "$PTBXL_ZIP" "$PTBXL_URL" || true
    elif command -v curl &> /dev/null; then
        curl -L -C - -o "$PTBXL_ZIP" "$PTBXL_URL" || true
    else
        echo "  ERROR: Neither wget nor curl found!"
        exit 1
    fi

    # Extract
    if [ -f "$PTBXL_ZIP" ]; then
        echo "  [EXTRACT] Unzipping..."
        unzip -o -q "$PTBXL_ZIP" -d "$DATA_DIR" || true
        rm -f "$PTBXL_ZIP"
    fi

    # Verify
    if verify_dataset; then
        echo "  [OK] Download and verification successful!"
        break
    else
        echo "  [FAIL] Verification failed, retrying..."
        attempt=$((attempt + 1))
    fi
done

if ! verify_dataset; then
    echo "  FATAL: Could not download/verify dataset after $MAX_RETRIES attempts!"
    exit 1
fi


# ==========================================================================
# Step 2: Virtual Environment Setup
# ==========================================================================

echo ""
echo "[STEP 2/5] Virtual Environment Setup"
echo "--------------------------------------"

if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating virtual environment in $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
else
    echo "  Virtual environment already exists."
fi

# Activate
source "$VENV_DIR/bin/activate"
echo "  Activated: $(which python)"
echo "  Python version: $(python --version)"


# ==========================================================================
# Step 3: Install Dependencies
# ==========================================================================

echo ""
echo "[STEP 3/5] Installing Dependencies"
echo "-------------------------------------"

pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (auto-detect)
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    echo "  CUDA already available via existing PyTorch."
else
    echo "  Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Install project requirements
pip install -r requirements.txt

# Verify GPU
echo ""
echo "  GPU Check:"
python -c "
import torch
print(f'    PyTorch: {torch.__version__}')
print(f'    CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    GPU: {torch.cuda.get_device_name(0)}')
    print(f'    VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

try:
    import pennylane as qml
    print(f'    PennyLane: {qml.__version__}')
except:
    print('    PennyLane: NOT INSTALLED')

try:
    dev = qml.device('lightning.gpu', wires=2)
    print(f'    Lightning GPU: AVAILABLE')
except:
    try:
        dev = qml.device('lightning.qubit', wires=2)
        print(f'    Lightning Qubit: AVAILABLE (GPU unavailable)')
    except:
        print(f'    Quantum: default.qubit only')
"


# ==========================================================================
# Step 4: Run Pipeline
# ==========================================================================

echo ""
echo "[STEP 4/5] Running Training Pipeline"
echo "--------------------------------------"

python pipeline.py \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --models "$MODELS"


# ==========================================================================
# Step 5: Git Commit & Push
# ==========================================================================

echo ""
echo "[STEP 5/5] Pushing Results to GitHub"
echo "--------------------------------------"

# Add all results
git add -A

# Commit with timestamp
COMMIT_MSG="Server run results — $(date '+%Y-%m-%d %H:%M:%S') — ${EPOCHS} epochs"
git commit -m "$COMMIT_MSG" || echo "  Nothing to commit."

# Push
git push || echo "  WARNING: git push failed. You may need to push manually."


# ==========================================================================
# Done
# ==========================================================================

echo ""
echo "======================================================================"
echo "  PIPELINE COMPLETE!"
echo "  Finished: $(date)"
echo "  Results:  $OUTPUT_DIR/"
echo "======================================================================"
