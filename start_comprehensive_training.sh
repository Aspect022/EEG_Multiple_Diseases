#!/bin/bash
# =============================================================================
# SNN Comprehensive Training - Simple Start Script
# =============================================================================
# Usage:
#   bash start_comprehensive_training.sh
#
# This creates all necessary directories and starts the training.
# =============================================================================

set -euo pipefail

# Create ALL necessary directories FIRST
mkdir -p logs
mkdir -p outputs/results
mkdir -p data

echo "======================================================================"
echo "  Starting Comprehensive SNN Training Pipeline"
echo "  Directories created: logs/, outputs/results/, data/"
echo "======================================================================"

# Export configuration
export CUDA_VISIBLE_DEVICES=0
DATA_DIR="./data"
OUTPUT_DIR="./outputs/results"
EPOCHS=50
BATCH_SIZE=128
LR=3e-4

# Run comprehensive pipeline
python comprehensive_snn_pipeline.py \
    --stage all \
    --force-continue

echo "======================================================================"
echo "  PIPELINE COMPLETE!"
echo "======================================================================"
