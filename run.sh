#!/bin/bash
# ==========================================================================
# EEG Sleep Staging Pipeline — Automated Server Deployment Script
# ==========================================================================
#
# This script fully automates the research pipeline on a GPU server:
#   1. Installs awscli and downloads BOAS ds005555 from OpenNeuro S3
#   2. Creates a Python virtual environment
#   3. Installs all dependencies (including GPU-accelerated PennyLane)
#   4. Runs all 19 experiments sequentially
#   5. Commits and pushes results to GitHub
#
# Usage:
#   bash run.sh                     # Full run (30 epochs)
#   bash run.sh --epochs 5          # Quick test run
#   bash run.sh --models snn_lif_resnet,swin  # Specific models only
#   bash run.sh --max-subjects 5    # Quick test with 5 subjects per split
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
MAX_SUBJECTS="${MAX_SUBJECTS:-}"

# Parse CLI args (override defaults)
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --max-subjects) MAX_SUBJECTS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; shift ;;
    esac
done

echo "======================================================================"
echo "  EEG SLEEP STAGING PIPELINE — AUTOMATED SERVER RUN"
echo "  Dataset: BOAS ds005555 (W/N1/N2/N3/REM)"
echo "======================================================================"
echo "  Epochs:       $EPOCHS"
echo "  Batch Size:   $BATCH_SIZE"
echo "  LR:           $LR"
echo "  Data Dir:     $DATA_DIR"
echo "  Output Dir:   $OUTPUT_DIR"
echo "  Models:       $MODELS"
echo "  Max Subjects: ${MAX_SUBJECTS:-all}"
echo "  Started:      $(date)"
echo "======================================================================"


# ==========================================================================
# Step 1: Dataset Download via AWS S3
# ==========================================================================

echo ""
echo "[STEP 1/5] BOAS Dataset Download from OpenNeuro S3"
echo "---------------------------------------------------"

BOAS_S3="s3://openneuro.org/ds005555"
BOAS_LOCAL="$DATA_DIR/ds005555"

# Install awscli if not available
if ! command -v aws &> /dev/null; then
    echo "  [INSTALL] Installing awscli..."
    pip install awscli --quiet
fi

# Verify AWS CLI is now available
if ! command -v aws &> /dev/null; then
    echo "  ERROR: awscli installation failed!"
    exit 1
fi
echo "  [OK] awscli: $(aws --version 2>&1 | head -1)"

# Download with S3 sync (parallel, resumable, no auth needed)
verify_dataset() {
    if [ ! -d "$BOAS_LOCAL" ]; then
        return 1
    fi
    local sub_count
    sub_count=$(find "$BOAS_LOCAL" -maxdepth 1 -type d -name "sub-*" 2>/dev/null | wc -l)
    if [ "$sub_count" -ge 10 ]; then
        local edf_count
        edf_count=$(find "$BOAS_LOCAL" -name "*.edf" 2>/dev/null | head -5 | wc -l)
        if [ "$edf_count" -ge 1 ]; then
            echo "  [OK] Dataset verified: $sub_count subjects, EDF files present"
            return 0
        fi
    fi
    return 1
}

mkdir -p "$DATA_DIR"

MAX_RETRIES=3
attempt=1

while [ $attempt -le $MAX_RETRIES ]; do
    if verify_dataset; then
        break
    fi

    echo "  [DOWNLOAD] Attempt $attempt/$MAX_RETRIES — aws s3 sync from OpenNeuro..."
    aws s3 sync --no-sign-request "$BOAS_S3" "$BOAS_LOCAL" || true

    if verify_dataset; then
        echo "  [OK] Download complete!"
        break
    else
        echo "  [RETRY] Verification failed, retrying..."
        attempt=$((attempt + 1))
    fi
done

if ! verify_dataset; then
    echo "  FATAL: Could not download/verify BOAS dataset after $MAX_RETRIES attempts!"
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

# Verify GPU + dependencies
echo ""
echo "  System Check:"
python -c "
import torch
print(f'    PyTorch: {torch.__version__}')
print(f'    CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    GPU: {torch.cuda.get_device_name(0)}')
    print(f'    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

try:
    import mne
    print(f'    MNE: {mne.__version__}')
except:
    print('    MNE: NOT INSTALLED')

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
echo "[STEP 4/5] Running EEG Training Pipeline"
echo "------------------------------------------"

PIPELINE_ARGS="--epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LR --data-dir $DATA_DIR --output-dir $OUTPUT_DIR --models $MODELS"

if [ -n "$MAX_SUBJECTS" ]; then
    PIPELINE_ARGS="$PIPELINE_ARGS --max-subjects $MAX_SUBJECTS"
fi

python pipeline.py $PIPELINE_ARGS


# ==========================================================================
# Step 5: Git Commit & Push
# ==========================================================================

echo ""
echo "[STEP 5/5] Pushing Results to GitHub"
echo "--------------------------------------"

# Add all results (but not the huge dataset)
echo "ds005555/" >> .gitignore 2>/dev/null || true
echo "data/" >> .gitignore 2>/dev/null || true

git add -A

# Commit with timestamp
COMMIT_MSG="EEG results — $(date '+%Y-%m-%d %H:%M:%S') — ${EPOCHS} epochs — BOAS ds005555"
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
