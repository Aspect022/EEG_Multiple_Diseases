#!/bin/bash
# ==========================================================================
# Paper-Focused BOAS Runner for Linux A100 Server
# ==========================================================================
#
# Uses the same venv convention as run.sh and launches only the cleaned,
# paper-relevant experiments from paper_pipeline.py.
#
# Usage:
#   chmod +x run_paper_pipeline.sh
#   nohup ./run_paper_pipeline.sh > logs/paper_pipeline.log 2>&1 &
#
# Optional overrides:
#   PRESET=core|extended|fusion_ablation
#   EPOCHS=50
#   BATCH_SIZE=128
#   LR=0.001
#   DATA_DIR=/path/to/data
#   OUTPUT_DIR=outputs/paper_runs
#   VENV_DIR=venv
#   MAX_SUBJECTS=10
#
# ==========================================================================

set -euo pipefail

PRESET="${PRESET:-core}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.001}"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/paper_runs}"
VENV_DIR="${VENV_DIR:-venv}"
PYTHON="${PYTHON:-python3}"
MAX_SUBJECTS="${MAX_SUBJECTS:-}"
NO_PRETRAINED="${NO_PRETRAINED:-0}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --preset) PRESET="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max-subjects) MAX_SUBJECTS="$2"; shift 2 ;;
        --venv-dir) VENV_DIR="$2"; shift 2 ;;
        --no-pretrained) NO_PRETRAINED=1; shift ;;
        *) echo "Unknown arg: $1"; shift ;;
    esac
done

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "======================================================================"
echo "  PAPER-FOCUSED EEG SLEEP STAGING RUN"
echo "======================================================================"
echo "  Preset:       $PRESET"
echo "  Epochs:       $EPOCHS"
echo "  Batch Size:   $BATCH_SIZE"
echo "  LR:           $LR"
echo "  Data Dir:     $DATA_DIR"
echo "  Output Dir:   $OUTPUT_DIR"
echo "  Max Subjects: ${MAX_SUBJECTS:-all}"
echo "  Pretrained:   $([ "$NO_PRETRAINED" -eq 1 ] && echo "disabled" || echo "enabled")"
echo "  Started:      $(date)"
echo "======================================================================"

# ==========================================================================
# Step 1: Activate the same venv convention as the main run.sh
# ==========================================================================
if [ ! -d "$VENV_DIR" ]; then
    echo "  [ERROR] Virtual environment not found at $VENV_DIR"
    echo "  Create it with the main setup flow first."
    exit 1
fi

source "$VENV_DIR/bin/activate"
echo "  [VENV] Activated: $(which python)"
echo "  [VENV] Python:    $(python --version)"

# ==========================================================================
# Step 2: Server runtime environment
# ==========================================================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export PYTHONWARNINGS="ignore::RuntimeWarning"

if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    echo "  [GPU] $(python -c "import torch; print(torch.cuda.get_device_name(0))")"
else
    echo "  [WARN] CUDA not visible from the active environment"
fi

# W&B is optional; training should continue if not configured.
if python -c "import wandb" 2>/dev/null; then
    echo "  [W&B] Available in environment"
else
    echo "  [W&B] Not installed in environment"
fi

# ==========================================================================
# Step 3: Run the paper-focused pipeline
# ==========================================================================
PIPELINE_ARGS=(
    --preset "$PRESET"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --data-dir "$DATA_DIR"
    --output-dir "$OUTPUT_DIR"
)

if [ -n "$MAX_SUBJECTS" ]; then
    PIPELINE_ARGS+=(--max-subjects "$MAX_SUBJECTS")
fi

if [ "$NO_PRETRAINED" -eq 1 ]; then
    PIPELINE_ARGS+=(--no-pretrained)
fi

echo ""
echo "  [RUN] python paper_pipeline.py ${PIPELINE_ARGS[*]}"
python paper_pipeline.py "${PIPELINE_ARGS[@]}"

echo ""
echo "======================================================================"
echo "  PAPER RUN COMPLETE"
echo "  Finished: $(date)"
echo "  Results:  $OUTPUT_DIR"
echo "======================================================================"
