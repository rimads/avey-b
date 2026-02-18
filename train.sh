#!/bin/bash

# --- CONFIGURATION ---
# Set default batch size here.
# Using ":-" allows you to override it via env vars (e.g., BATCH_SIZE=64 ./train.sh)
BATCH_SIZE="${BATCH_SIZE:-32}"

# --- AUTOMATIC GPU DETECTION ---
# We use nvidia-smi to list GPUs and wc -l to count the lines.
if command -v nvidia-smi &> /dev/null; then
    NUMBER_OF_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    echo "Warning: nvidia-smi not found. Defaulting to 1."
    NUMBER_OF_GPUS=1
fi

# Trim whitespace from the count
NUMBER_OF_GPUS=$(echo "$NUMBER_OF_GPUS" | xargs)

# --- EXECUTION LOGIC ---
echo "Configuration: BATCH_SIZE=$BATCH_SIZE | GPUS=$NUMBER_OF_GPUS"

if [ "$NUMBER_OF_GPUS" -le 1 ]; then
    # CASE 1: Single GPU (or CPU) -> Run standard Python
    echo "Starting Single-GPU run..."
    python -m train_mlm \
        --device_bsz "$BATCH_SIZE" \
        "$@"
else
    # CASE 2: Multi-GPU -> Run Torchrun
    echo "Starting Distributed run with torchrun..."
    torchrun \
        --nproc_per_node="$NUMBER_OF_GPUS" \
        --nnodes=1 \
        -m train_mlm \
        --device_bsz "$BATCH_SIZE" \
        "$@"
fi
