#!/usr/bin/env bash
# =============================================================================
# run_final.sh  —  EEG Sleep-EDF Final Training Launcher
#
# Does everything automatically:
#   1. Finds and activates the project virtualenv
#   2. Installs / updates missing Python dependencies
#   3. Downloads Sleep-EDF dataset if not present
#   4. Validates all 13 model forward passes
#   5. Launches training via nohup (runs even after you disconnect)
#
# Usage:
#   bash run_final.sh                    # run all 13 models
#   bash run_final.sh --models phase1    # Phase-1 only (1D models)
#   bash run_final.sh --models phase2    # Phase-2 only (2D/fusion)
#   bash run_final.sh --models snn_1d_attn,conditional_routing
#   bash run_final.sh --epochs 50 --batch-size 32
#
# Watch the log afterwards:
#   tail -f logs/sleep_edf_run.log
# =============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
DATA_DIR="${SCRIPT_DIR}/data/sleep-edf"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/sleep_edf_results"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/sleep_edf_run.log"
PIPELINE="${SCRIPT_DIR}/sleep_edf_pipeline.py"

# ── Defaults (override via args) ─────────────────────────────────────────────
MODELS="all"
EPOCHS=30
BATCH_SIZE=64
NUM_WORKERS=4
LR="1e-3"
SKIP_DOWNLOAD=0
SKIP_VALIDATE=0

# ── Parse extra args to pass through ─────────────────────────────────────────
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)        MODELS="$2";      shift 2 ;;
        --epochs)        EPOCHS="$2";      shift 2 ;;
        --batch-size)    BATCH_SIZE="$2";  shift 2 ;;
        --num-workers)   NUM_WORKERS="$2"; shift 2 ;;
        --lr)            LR="$2";          shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=1;  shift   ;;
        --skip-validate) SKIP_VALIDATE=1;  shift   ;;
        *)               EXTRA_ARGS+=("$1"); shift  ;;
    esac
done

# ── Color helpers ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; exit 1; }
step()  { echo -e "\n${GREEN}══${NC} $* ${GREEN}══${NC}"; }

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       EEG Sleep-EDF Final Training  —  run_final.sh         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Models:      ${MODELS}"
echo "  Epochs:      ${EPOCHS}"
echo "  Batch size:  ${BATCH_SIZE}"
echo "  Workers:     ${NUM_WORKERS}"
echo "  LR:          ${LR}"
echo "  Log file:    ${LOG_FILE}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Find & activate virtualenv
# ─────────────────────────────────────────────────────────────────────────────
step "Step 1: Virtual environment"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    info "Found existing venv at ${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    PYTHON="${VENV_DIR}/bin/python3"
    PIP="${VENV_DIR}/bin/pip"
    info "Using Python: $(${PYTHON} --version)"
else
    warn "No venv found — creating one at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    PYTHON="${VENV_DIR}/bin/python3"
    PIP="${VENV_DIR}/bin/pip"
    info "Created venv. Python: $(${PYTHON} --version)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Install / update dependencies
# ─────────────────────────────────────────────────────────────────────────────
step "Step 2: Checking dependencies"

# Always upgrade pip silently
${PIP} install --upgrade pip --quiet

# Check which packages are missing and install only those
MISSING=()
check_pkg() {
    local pkg="$1" import_name="${2:-$1}"
    if ! ${PYTHON} -c "import ${import_name}" &>/dev/null 2>&1; then
        MISSING+=("${pkg}")
    fi
}

check_pkg "torch"
check_pkg "torchvision"
check_pkg "timm"
check_pkg "einops"
check_pkg "mne"
check_pkg "numpy"
check_pkg "pandas"
check_pkg "scikit-learn" "sklearn"
check_pkg "scipy"
check_pkg "wandb"
check_pkg "tqdm"
check_pkg "snntorch"
check_pkg "wfdb"
check_pkg "matplotlib"
check_pkg "PyYAML" "yaml"
check_pkg "fvcore"
check_pkg "thop"
check_pkg "Pillow" "PIL"

if [[ ${#MISSING[@]} -gt 0 ]]; then
    warn "Installing missing packages: ${MISSING[*]}"
    ${PIP} install --quiet "${MISSING[@]}"
    info "Dependencies installed."
else
    info "All required packages already installed."
fi

# Always ensure wandb is logged in (uses WANDB_API_KEY env var if set)
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    ${PYTHON} -c "import wandb; wandb.login(key='${WANDB_API_KEY}', relogin=True)" --quiet 2>/dev/null || true
    info "W&B logged in via WANDB_API_KEY."
else
    # Check if already logged in
    if ${PYTHON} -c "import wandb; wandb.Api()" &>/dev/null 2>&1; then
        info "W&B already authenticated."
    else
        warn "W&B not authenticated. Set WANDB_API_KEY or run: wandb login"
        warn "Training will continue but W&B logging will be disabled."
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Download Sleep-EDF dataset if missing
# ─────────────────────────────────────────────────────────────────────────────
step "Step 3: Dataset"

PHYSIONET_URL="https://physionet.org/files/sleep-edfx/1.0.0"
# Cassette group has 153 recordings (20,000+ epochs) — more than enough
CASSETTE_URL="${PHYSIONET_URL}/sleep-cassette"

count_pairs() {
    find "${DATA_DIR}" -name "*PSG.edf" 2>/dev/null | wc -l
}

if [[ "${SKIP_DOWNLOAD}" -eq 1 ]]; then
    info "Skipping download check (--skip-download)."
elif [[ $(count_pairs) -ge 10 ]]; then
    info "Dataset OK: $(count_pairs) PSG files found at ${DATA_DIR}."
else
    warn "Sleep-EDF dataset not found or incomplete at ${DATA_DIR}."
    warn "Found $(count_pairs)/10 minimum required PSG files."
    echo ""
    echo "  Downloading Sleep-EDF Cassette subset (153 recordings, ~4 GB)..."
    echo "  This will take several minutes depending on your connection."
    echo ""

    mkdir -p "${DATA_DIR}"

    # Prefer wget with resume support; fall back to curl
    if command -v wget &>/dev/null; then
        DOWNLOADER="wget -q --show-progress -c -P ${DATA_DIR} -r -nd -np -A '*.edf,*.csv'"
        wget -q --show-progress -c \
            -r -nd -np \
            -A "*.edf" \
            --directory-prefix="${DATA_DIR}" \
            "${CASSETTE_URL}/" 2>&1 | tail -5 || true
    elif command -v curl &>/dev/null; then
        # Download the index page first to get file list
        curl -s "${CASSETTE_URL}/" \
            | grep -oP '(?<=href=")[^"]+\.edf' \
            | head -60 \
            | while read -r fname; do
                curl -# -C - -O --output-dir "${DATA_DIR}" \
                    "${CASSETTE_URL}/${fname}" 2>/dev/null || true
            done
    else
        error "Neither wget nor curl found. Install one and retry."
    fi

    FOUND=$(count_pairs)
    if [[ "${FOUND}" -ge 10 ]]; then
        info "Download complete. ${FOUND} PSG files ready."
    else
        error "Download incomplete: only ${FOUND} PSG files found. " \
              "Try manually: wget -r -np -nd -A '*.edf' ${CASSETTE_URL}/"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Validate model forward passes
# ─────────────────────────────────────────────────────────────────────────────
step "Step 4: Model validation"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

if [[ "${SKIP_VALIDATE}" -eq 1 ]]; then
    info "Skipping model validation (--skip-validate)."
else
    info "Running forward-pass smoke test for selected models..."

    # Use a temp script to bypass the dataset check for validation
    VALIDATE_OUT=$( ${PYTHON} - <<PYEOF 2>&1
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from pipeline import EXPERIMENT_DEFS, create_model
import torch

selector = '${MODELS}'
if selector == 'all':
    from sleep_edf_pipeline import ALL_MODELS
    keys = ALL_MODELS
elif selector == 'phase1':
    from sleep_edf_pipeline import PHASE1_MODELS
    keys = PHASE1_MODELS
elif selector == 'phase2':
    from sleep_edf_pipeline import PHASE2_MODELS
    keys = PHASE2_MODELS
else:
    keys = [k.strip() for k in selector.split(',') if k.strip()]

failed = []
for key in keys:
    if key not in EXPERIMENT_DEFS:
        print(f'  SKIP {key:<30} not in EXPERIMENT_DEFS')
        continue
    cfg = EXPERIMENT_DEFS[key]
    try:
        model = create_model(cfg, num_classes=5)
        model.eval()
        data_mode = cfg.get('data_mode', '2d')
        with torch.no_grad():
            if data_mode == '1d':
                out = model(torch.randn(2, 6, 3000))
            elif data_mode == 'both':
                out = model(raw_signal=torch.randn(2, 6, 3000), scalogram=torch.randn(2, 3, 224, 224))
            else:
                out = model(torch.randn(2, 3, 224, 224))
        logits = out[0] if isinstance(out, (tuple, list)) else out
        assert logits.shape == (2, 5), f'Bad shape {logits.shape}'
        print(f'  OK   {key:<30} output={tuple(logits.shape)}  params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    except Exception as e:
        print(f'  FAIL {key:<30} {e}')
        failed.append(key)

if failed:
    print(f'\nFAILED models: {failed}')
    sys.exit(1)
else:
    print(f'\nAll models OK.')
PYEOF
    )

    echo "${VALIDATE_OUT}"

    if echo "${VALIDATE_OUT}" | grep -q "FAILED models:"; then
        error "Some models failed validation. Fix errors above before training."
    else
        info "All models validated successfully."
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Launch training with nohup
# ─────────────────────────────────────────────────────────────────────────────
step "Step 5: Launching training"

CMD="${PYTHON} ${PIPELINE} \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --lr ${LR} \
    --models ${MODELS} \
    ${EXTRA_ARGS[*]:-}"

info "Command: ${CMD}"
echo ""

# Clear old log or append with separator
if [[ -f "${LOG_FILE}" ]]; then
    echo "" >> "${LOG_FILE}"
    echo "═══════════════════════════════════════════════════════════════" >> "${LOG_FILE}"
    echo "  NEW RUN — $(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE}"
    echo "═══════════════════════════════════════════════════════════════" >> "${LOG_FILE}"
fi

nohup ${CMD} >> "${LOG_FILE}" 2>&1 &
PID=$!

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
printf  "║  Training started!  PID: %-5s                               ║\n" "${PID}"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Watch:   tail -f ${LOG_FILE}"
echo "║  Kill:    kill ${PID}"
echo "║  W&B:     https://wandb.ai  →  project: eeg-sleep-apnea"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "${PID}" > "${LOG_DIR}/sleep_edf_run.pid"
info "PID saved to ${LOG_DIR}/sleep_edf_run.pid"
