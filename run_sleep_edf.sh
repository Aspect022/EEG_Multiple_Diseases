#!/bin/bash
# ==========================================================================
# Sleep-EDF Benchmark Runner for Linux A100 Server
# ==========================================================================

set -euo pipefail

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-0.001}"
DATA_DIR="${DATA_DIR:-data/sleep-edf}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sleep_edf_results}"
VENV_DIR="${VENV_DIR:-venv}"
MODELS="${MODELS:-stable}"
MAX_RECORDS="${MAX_RECORDS:-}"
NO_PRETRAINED="${NO_PRETRAINED:-0}"
VALIDATE_FIRST="${VALIDATE_FIRST:-1}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --max-records) MAX_RECORDS="$2"; shift 2 ;;
        --venv-dir) VENV_DIR="$2"; shift 2 ;;
        --no-pretrained) NO_PRETRAINED=1; shift ;;
        --no-validate-first) VALIDATE_FIRST=0; shift ;;
        *) echo "Unknown arg: $1"; shift ;;
    esac
done

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "======================================================================"
echo "  SLEEP-EDF EEG BENCHMARK RUN"
echo "======================================================================"
echo "  Models:       $MODELS"
echo "  Epochs:       $EPOCHS"
echo "  Batch Size:   $BATCH_SIZE"
echo "  Num Workers:  $NUM_WORKERS"
echo "  LR:           $LR"
echo "  Data Dir:     $DATA_DIR"
echo "  Output Dir:   $OUTPUT_DIR"
echo "  Max Records:  ${MAX_RECORDS:-all}"
echo "======================================================================"

if [ ! -d "$VENV_DIR" ]; then
    echo "[ERROR] Virtual environment not found at $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export PYTHONWARNINGS="ignore::RuntimeWarning"

BASE_ARGS=(
    --models "$MODELS"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --data-dir "$DATA_DIR"
    --output-dir "$OUTPUT_DIR"
)

if [ -n "$MAX_RECORDS" ]; then
    BASE_ARGS+=(--max-records "$MAX_RECORDS")
fi

if [ "$NO_PRETRAINED" -eq 1 ]; then
    BASE_ARGS+=(--no-pretrained)
fi

if [ "$VALIDATE_FIRST" -eq 1 ]; then
    echo ""
    echo "[Validation] Running model compatibility check first..."
    python sleep_edf_pipeline.py "${BASE_ARGS[@]}" --validate-only
fi

echo ""
echo "[Run] Starting training..."
python sleep_edf_pipeline.py "${BASE_ARGS[@]}"
