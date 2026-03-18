#!/bin/bash
# =============================================================================
# SNN Multi-Modal Training Script for A100 GPU
# =============================================================================
# Usage: ./train_snn_fusion.sh
# 
# This script runs all SNN experiments in sequence:
# 1. Fixed 1D SNN (baseline)
# 2. Fixed 2D SNN (baseline)
# 3. Gated Fusion (main contribution)
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Paths
PROJECT_DIR="/path/to/your/EEG/project"  # UPDATE THIS!
DATA_DIR="/path/to/datasets/BOAS"        # UPDATE THIS!
OUTPUT_DIR="./outputs"
LOG_DIR="./logs"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# A100 Optimization
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# Training hyperparameters (A100 optimized)
BATCH_SIZE=128          # A100 can handle large batches
NUM_WORKERS=8           # A100 + fast CPU = more workers
EPOCHS=50               # SNNs need more epochs
LEARNING_RATE=3e-4      # SNN-optimized

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/training_master.log"
}

run_experiment() {
    local exp_name=$1
    local exp_type=$2
    
    log "=========================================="
    log "Starting: $exp_name ($exp_type)"
    log "=========================================="
    
    python pipeline.py \
        --experiment "$exp_name" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --learning-rate $LEARNING_RATE \
        --num-workers $NUM_WORKERS \
        2>&1 | tee -a "$LOG_DIR/${exp_name}.log"
    
    log "Completed: $exp_name"
    log "Results saved to: $OUTPUT_DIR/results/${exp_name}_results.json"
}

# =============================================================================
# Training Pipeline
# =============================================================================

log "=========================================="
log "SNN Multi-Modal Training Pipeline"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
log "CUDA: $(nvcc --version | grep release | cut -d',' -f1)"
log "=========================================="

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# =============================================================================
# Phase 1: Single Modality Baselines
# =============================================================================

log "PHASE 1: Single Modality Baselines"

# 1D SNN (raw EEG) - Fast, efficient
run_experiment "snn_1d_lif" "1D SNN baseline"

# 2D SNN (scalograms) - Higher accuracy
run_experiment "snn" "2D SNN baseline"

# =============================================================================
# Phase 2: Fusion Models (Main Contribution)
# =============================================================================

log "PHASE 2: Fusion Models"

# Early Fusion - Maximum accuracy
run_experiment "snn_fusion_early" "Early fusion baseline"

# Late Fusion - Ensemble, interpretable
run_experiment "snn_fusion_late" "Late fusion baseline"

# Gated Fusion - NOVEL: Confidence-based routing (recommended!)
run_experiment "snn_fusion_gated" "Gated fusion (main contribution)"

# =============================================================================
# Summary
# =============================================================================

log "=========================================="
log "ALL TRAINING COMPLETE!"
log "=========================================="
log "Results directory: $OUTPUT_DIR/results/"
log "Logs directory: $LOG_DIR/"
log ""
log "To view results:"
log "  python analyze_results.py"
log ""
log "To monitor training:"
log "  tail -f $LOG_DIR/snn_fusion_gated.log"
log "=========================================="
