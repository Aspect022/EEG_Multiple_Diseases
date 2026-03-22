#!/bin/bash
# ==========================================================================
# Sleep-EDF Benchmark Runner - Automated Linux Server Script
# ==========================================================================
#
# This script mirrors the BOAS automation flow:
#   1. Creates / activates a virtual environment
#   2. Installs PyTorch + project dependencies
#   3. Downloads Sleep-EDF Expanded from PhysioNet if missing
#   4. Validates model compatibility
#   5. Starts Sleep-EDF training
#
# Usage:
#   chmod +x run_sleep_edf.sh
#   nohup ./run_sleep_edf.sh > logs/sleep_edf.log 2>&1 &
#
# Optional overrides:
#   EPOCHS=30
#   BATCH_SIZE=64
#   NUM_WORKERS=8
#   LR=0.001
#   DATA_DIR=data/sleep-edf
#   OUTPUT_DIR=outputs/sleep_edf_results
#   VENV_DIR=venv
#   MODELS=stable
#   TORCH_CHANNEL=cu121
#   INSTALL_OPTIONAL_QUANTUM=0
#   VALIDATE_FIRST=1
#   MAX_RECORDS=10
#
# ==========================================================================

set -euo pipefail

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-0.001}"
DATA_DIR="${DATA_DIR:-data/sleep-edf}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sleep_edf_results}"
VENV_DIR="${VENV_DIR:-venv}"
PYTHON="${PYTHON:-python3}"
MODELS="${MODELS:-stable}"
MAX_RECORDS="${MAX_RECORDS:-}"
TORCH_CHANNEL="${TORCH_CHANNEL:-cu121}"
INSTALL_OPTIONAL_QUANTUM="${INSTALL_OPTIONAL_QUANTUM:-0}"
VALIDATE_FIRST="${VALIDATE_FIRST:-1}"
NO_PRETRAINED="${NO_PRETRAINED:-0}"
SLEEP_EDF_URL="${SLEEP_EDF_URL:-https://physionet.org/files/sleep-edfx/1.0.0/}"
DOWNLOAD_RETRY_SLEEP="${DOWNLOAD_RETRY_SLEEP:-60}"
MAX_DOWNLOAD_ATTEMPTS="${MAX_DOWNLOAD_ATTEMPTS:-0}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --venv-dir) VENV_DIR="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --max-records) MAX_RECORDS="$2"; shift 2 ;;
        --torch-channel) TORCH_CHANNEL="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --install-optional-quantum) INSTALL_OPTIONAL_QUANTUM=1; shift ;;
        --no-validate-first) VALIDATE_FIRST=0; shift ;;
        --no-pretrained) NO_PRETRAINED=1; shift ;;
        *) echo "Unknown arg: $1"; shift ;;
    esac
done

mkdir -p logs
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"

echo "======================================================================"
echo "  SLEEP-EDF EEG BENCHMARK - AUTOMATED SERVER RUN"
echo "======================================================================"
echo "  Models:       $MODELS"
echo "  Epochs:       $EPOCHS"
echo "  Batch Size:   $BATCH_SIZE"
echo "  Num Workers:  $NUM_WORKERS"
echo "  LR:           $LR"
echo "  Data Dir:     $DATA_DIR"
echo "  Output Dir:   $OUTPUT_DIR"
echo "  Max Records:  ${MAX_RECORDS:-all}"
echo "  Torch Wheel:  $TORCH_CHANNEL"
echo "  Venv:         $VENV_DIR"
echo "  DL Retry:     ${DOWNLOAD_RETRY_SLEEP}s"
echo "  DL Attempts:  ${MAX_DOWNLOAD_ATTEMPTS} (0=infinite)"
echo "  Started:      $(date)"
echo "======================================================================"


# ==========================================================================
# Step 1: Virtual Environment
# ==========================================================================

echo ""
echo "[STEP 1/5] Virtual Environment Setup"
echo "------------------------------------"

if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating virtual environment in $VENV_DIR..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "  Virtual environment already exists."
fi

source "$VENV_DIR/bin/activate"
echo "  Activated: $(which python)"
echo "  Python:    $(python --version)"


# ==========================================================================
# Step 2: Install Dependencies
# ==========================================================================

echo ""
echo "[STEP 2/5] Installing Dependencies"
echo "----------------------------------"

pip install --upgrade pip setuptools wheel

if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    echo "  CUDA already available via existing PyTorch."
else
    echo "  Installing PyTorch ($TORCH_CHANNEL)..."
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$TORCH_CHANNEL"
fi

pip install -r requirements.txt

if [ "$INSTALL_OPTIONAL_QUANTUM" -eq 1 ]; then
    pip install -r requirements_optional_quantum.txt || true
fi

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
    import timm
    print(f'    timm: {timm.__version__}')
except Exception:
    print('    timm: NOT INSTALLED')
try:
    import snntorch
    print(f'    snntorch: {snntorch.__version__}')
except Exception:
    print('    snntorch: NOT INSTALLED')
try:
    import mne
    print(f'    MNE: {mne.__version__}')
except Exception:
    print('    MNE: NOT INSTALLED')
"


# ==========================================================================
# Step 3: Download Sleep-EDF
# ==========================================================================

echo ""
echo "[STEP 3/5] Sleep-EDF Download"
echo "-----------------------------"

verify_dataset() {
    if python - <<PY
from src.data.sleep_edf_dataset import MIN_RECOMMENDED_RECORD_PAIRS, verify_sleep_edf_dataset
import sys

data_dir = r"""$DATA_DIR"""
if verify_sleep_edf_dataset(data_dir):
    print(f"  [OK] Sleep-EDF present with at least {MIN_RECOMMENDED_RECORD_PAIRS} matched pairs")
    sys.exit(0)
sys.exit(1)
PY
    then
        return 0
    fi
    return 1
}

report_sleep_edf_status() {
    python - <<PY
from pathlib import Path
from src.data.sleep_edf_dataset import _discover_record_pairs, MIN_RECOMMENDED_RECORD_PAIRS

root = Path(r"""$DATA_DIR""")
pairs = _discover_record_pairs(root) if root.exists() else []
psg_count = len(list(root.rglob("*PSG.edf"))) if root.exists() else 0
hyp_count = len(list(root.rglob("*Hypnogram.edf"))) if root.exists() else 0
subjects = len({pair["subject_id"] for pair in pairs})
print(
    f"  Current dataset status: {len(pairs)} matched pairs, "
    f"{subjects} subjects, {psg_count} PSG, {hyp_count} hypnograms "
    f"(recommended minimum pairs: {MIN_RECOMMENDED_RECORD_PAIRS})"
)
PY
}

download_with_wget() {
    echo "  Downloading from PhysioNet: $SLEEP_EDF_URL"
    wget \
        -r -N -c -np -nH --cut-dirs=3 \
        -R "index.html*" \
        --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=30 -t 0 \
        "$SLEEP_EDF_URL" \
        -P "$DATA_DIR"
}

retry_until_complete() {
    local attempt=1

    while true; do
        if verify_dataset; then
            echo "  Dataset ready."
            return 0
        fi

        report_sleep_edf_status
        if [ "$MAX_DOWNLOAD_ATTEMPTS" -gt 0 ] && [ "$attempt" -gt "$MAX_DOWNLOAD_ATTEMPTS" ]; then
            echo "  FATAL: Reached max download attempts ($MAX_DOWNLOAD_ATTEMPTS) before dataset became ready."
            return 1
        fi

        echo "  Download attempt $attempt..."
        if download_with_wget; then
            echo "  Download attempt $attempt finished."
        else
            echo "  Download attempt $attempt ended with wget errors; will retry after ${DOWNLOAD_RETRY_SLEEP}s."
        fi

        if verify_dataset; then
            echo "  Dataset ready after download attempt $attempt."
            return 0
        fi

        report_sleep_edf_status
        echo "  Dataset still incomplete. Sleeping ${DOWNLOAD_RETRY_SLEEP}s before retry..."
        sleep "$DOWNLOAD_RETRY_SLEEP"
        attempt=$((attempt + 1))
    done
}

if ! retry_until_complete; then
    echo "  FATAL: Sleep-EDF download/verification failed."
    exit 1
fi


# ==========================================================================
# Step 4: Validate
# ==========================================================================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export PYTHONWARNINGS="ignore::RuntimeWarning"

PIPELINE_ARGS=(
    --models "$MODELS"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --data-dir "$DATA_DIR"
    --output-dir "$OUTPUT_DIR"
)

if [ -n "$MAX_RECORDS" ]; then
    PIPELINE_ARGS+=(--max-records "$MAX_RECORDS")
fi

if [ "$NO_PRETRAINED" -eq 1 ]; then
    PIPELINE_ARGS+=(--no-pretrained)
fi

if [ "$VALIDATE_FIRST" -eq 1 ]; then
    echo ""
    echo "[STEP 4/5] Validation"
    echo "---------------------"
    python sleep_edf_pipeline.py "${PIPELINE_ARGS[@]}" --validate-only
fi


# ==========================================================================
# Step 5: Train
# ==========================================================================

echo ""
echo "[STEP 5/5] Training"
echo "-------------------"
echo "  [RUN] python sleep_edf_pipeline.py ${PIPELINE_ARGS[*]}"
python sleep_edf_pipeline.py "${PIPELINE_ARGS[@]}"

echo ""
echo "======================================================================"
echo "  SLEEP-EDF RUN COMPLETE"
echo "  Finished: $(date)"
echo "  Results:  $OUTPUT_DIR"
echo "======================================================================"
