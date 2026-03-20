#!/bin/bash
# ============================================================================
# Sleep Apnea Classification - Training Launcher
# ============================================================================
# Usage:
#   ./run_sleep_apnea.sh                  # Run all experiments
#   ./run_sleep_apnea.sh --model cnn      # Run specific model
#   ./run_sleep_apnea.sh --ssl-pretrain   # With self-supervised pretraining
# ============================================================================

set -e

# Configuration
LOG_DIR="logs/sleep_apnea"
OUTPUT_DIR="outputs/sleep_apnea"
DATA_DIR="/home/user04/data/shhs"  # Update this path

# Create directories
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

# Parse arguments
MODEL="all"
SSL_PRETRAIN=""
EPOCHS=30
BATCH_SIZE=32

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --ssl-pretrain)
            SSL_PRETRAIN="--ssl-pretrain"
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL       Model to run (cnn, resnet18, vit_bilstm, all)"
            echo "  --ssl-pretrain      Enable self-supervised pretraining"
            echo "  --epochs N          Number of training epochs"
            echo "  --batch-size N      Batch size"
            echo "  --data-dir PATH     Path to dataset"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/apnea_${MODEL}_${TIMESTAMP}.log"

echo "============================================================================"
echo "  SLEEP APNEA CLASSIFICATION - TRAINING"
echo "============================================================================"
echo "  Model:        $MODEL"
echo "  Epochs:       $EPOCHS"
echo "  Batch Size:   $BATCH_SIZE"
echo "  Data Dir:     $DATA_DIR"
echo "  SSL Pretrain: ${SSL_PRETRAIN:-No}"
echo "  Log File:     $LOG_FILE"
echo "============================================================================"
echo ""

# Check GPU
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "⚠️  WARNING: Data directory not found: $DATA_DIR"
    echo "   Please download the dataset first:"
    echo "   - SHHS: https://sleepdata.org/datasets/shhs"
    echo "   - Apnea-ECG: https://physionet.org/content/apnea-ecg/"
    echo ""
    echo "   For testing, you can use:"
    echo "   wget -r -N -c -np https://physionet.org/files/apnea-ecg/1.0.0/ -P data/"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start training
echo "Starting training at $(date)"
echo "Logging to: $LOG_FILE"
echo ""

nohup python3 run_sleep_apnea_experiments.py \
    --model $MODEL \
    --ssl-pretrain $SSL_PRETRAIN \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --data-dir $DATA_DIR \
    > $LOG_FILE 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop training:"
echo "  kill $PID"
echo ""
echo "============================================================================"

# Wait a bit and show initial output
sleep 5
echo "Initial output:"
tail -20 $LOG_FILE
