#!/bin/bash
# =============================================================================
# SNN Comprehensive Training - NoHup Command
# =============================================================================
# Usage:
#   nohup bash run_comprehensive_snn.sh > logs/comprehensive_snn.log 2>&1 &
#
# This runs ALL SNN variants in sequence:
# 1. 1D SNNs (LIF, QIF, with/without attention)
# 2. 2D SNNs (LIF, QIF, ResNet, ViT)
# 3. Original Fusion (fusion_b, fusion_c)
# 4. New SNN Fusion (Early, Late, Gated)
# 5. Quantum-SNN Fusion (RXY-full + SNN) ⭐ NOVEL
# =============================================================================

set -euo pipefail

# Configuration (from run.sh)
export CUDA_VISIBLE_DEVICES=0
DATA_DIR="./data"
OUTPUT_DIR="./outputs/results"
LOG_DIR="./logs"
EPOCHS=50
BATCH_SIZE=128
LR=3e-4
NUM_WORKERS=8

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

echo "======================================================================"
echo "  COMPREHENSIVE SNN TRAINING PIPELINE"
echo "  Started: $(date)"
echo "======================================================================"
echo "  Data:     $DATA_DIR"
echo "  Output:   $OUTPUT_DIR"
echo "  Logs:     $LOG_DIR"
echo "  Epochs:   $EPOCHS"
echo "  Batch:    $BATCH_SIZE"
echo "  LR:       $LR"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "======================================================================"

# Run comprehensive pipeline
python3 comprehensive_snn_pipeline.py \
    --stage all \
    --force-continue

echo ""
echo "======================================================================"
echo "  PIPELINE COMPLETE!"
echo "  Finished: $(date)"
echo "  Results:  $OUTPUT_DIR/"
echo "======================================================================"
