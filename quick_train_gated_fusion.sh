#!/bin/bash
# =============================================================================
# Quick Start: Gated Fusion Training (Single Command)
# =============================================================================
# This is the simplest command to run the most important experiment.
# 
# Usage on server:
#   chmod +x quick_train_gated_fusion.sh
#   nohup ./quick_train_gated_fusion.sh > logs/gated_fusion.log 2>&1 &
# =============================================================================

# Update these paths!
DATA_DIR="/path/to/datasets/BOAS"
OUTPUT_DIR="./outputs"

# A100 optimizations
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run gated fusion (the main contribution!)
python pipeline.py \
    --experiment snn_fusion_gated \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 50 \
    --batch-size 128 \
    --learning-rate 3e-4 \
    --num-workers 8 \
    2>&1 | tee -a logs/gated_fusion.log
