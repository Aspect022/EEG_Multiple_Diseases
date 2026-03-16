#!/bin/bash
# ==========================================================================
# Phase 2: 1D SNN + Quantum-1D + Fusion — nohup runner
# ==========================================================================
#
# Usage (on server):
#   nohup bash run_phase2.sh > phase2.log 2>&1 &
#   tail -f phase2.log
#
# ==========================================================================

set -euo pipefail

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"

echo "======================================================================"
echo "  PHASE 2: Raw 1D Signal + Fusion Architectures"
echo "  Total: 20 new models (3 SNN-1D + 14 Quantum-1D + 3 Fusion)"
echo "  Epochs: $EPOCHS | Batch: $BATCH_SIZE"
echo "  Started: $(date)"
echo "======================================================================"

# ── Activate venv ──
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "  [VENV] Activated: $(which python)"
else
    echo "  [ERROR] No venv found. Run 'bash run.sh' first to set up."
    exit 1
fi

# ── W&B setup ──
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export PYTHONWARNINGS="ignore::RuntimeWarning"

# ── Common args ──
COMMON="--skip-download --epochs $EPOCHS --batch-size $BATCH_SIZE"

# ==========================================================================
# Step 1: SNN-1D (prove SNN works on raw 1D signals)
# ==========================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1/3: SNN-1D Models (3 models)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python pipeline.py $COMMON --models snn_1d_lif,snn_1d_attn,spiking_vit_1d

# ==========================================================================
# Step 2: Quantum-1D (14 rotation × entanglement variants)
# ==========================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2/3: Quantum-1D Models (14 models)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

Q1D_MODELS="quantum_1d_ring_RX,quantum_1d_ring_RY,quantum_1d_ring_RZ"
Q1D_MODELS="$Q1D_MODELS,quantum_1d_ring_RXY,quantum_1d_ring_RXZ,quantum_1d_ring_RYZ,quantum_1d_ring_RXYZ"
Q1D_MODELS="$Q1D_MODELS,quantum_1d_full_RX,quantum_1d_full_RY,quantum_1d_full_RZ"
Q1D_MODELS="$Q1D_MODELS,quantum_1d_full_RXY,quantum_1d_full_RXZ,quantum_1d_full_RYZ,quantum_1d_full_RXYZ"

python pipeline.py $COMMON --models $Q1D_MODELS

# ==========================================================================
# Step 3: Fusion Models (3 models)
# ==========================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3/3: Fusion Models (3 models)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# NOTE: fusion_c (Multi-Modal) is excluded — it requires dual inputs
# (raw_signal + scalogram) which needs a custom training loop.
# We'll implement that separately.
python pipeline.py $COMMON --models fusion_a,fusion_b

# ==========================================================================
# Git Push Results
# ==========================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Pushing results to GitHub..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

git add -A
git commit -m "Phase 2 results — $(date '+%Y-%m-%d %H:%M:%S') — ${EPOCHS}ep — SNN-1D + Q-1D + Fusion" || echo "  Nothing to commit."
git push || echo "  WARNING: git push failed."

echo ""
echo "======================================================================"
echo "  PHASE 2 COMPLETE!"
echo "  Finished: $(date)"
echo "  Results:  outputs/results/"
echo "======================================================================"
