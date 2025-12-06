#!/bin/bash
# Fast VideoMME Evaluation with vLLM
# Expected: ~20 seconds per sample instead of 360 seconds (18x speedup!)

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

DATA_DIR="/work/u1399652/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/video"
ANNO_PATH="/work/u1399652/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/test-00000-of-00001.parquet"
OUTPUT="./results/videomme_vllm/results.json"
CACHE_DIR="./cache/videomme_vllm"

# Use batch size for maximum throughput
BATCH_SIZE=24
MAX_FRAMES=4
NUM_GPUS=7  # Tensor parallelism across 7 GPUs

# Optional: Use subset for faster iteration
SUBSET=0.33  # Uncomment to run on 1/3 of dataset (proportionally sampled from each duration category)

echo "Starting Fast VideoMME Evaluation with vLLM"
echo "==========================================="
echo "Batch size: $BATCH_SIZE"
echo "Max frames: $MAX_FRAMES"
echo "GPUs: $NUM_GPUS (tensor parallelism)"
if [ ! -z "$SUBSET" ]; then
    echo "Using subset: ${SUBSET} of full dataset"
    echo "Expected time: ~$(python3 -c "print(int(2.5 * $SUBSET))") hour(s)"
else
    echo "Expected time: ~2-3 hours (vs 11+ hours without vLLM)"
fi
echo ""

# Create output directory
mkdir -p "$(dirname "$OUTPUT")"

# Build command with optional subset parameter
CMD="python src/eval/evaluate_videomme_vllm.py \
    --data_dir \"$DATA_DIR\" \
    --anno_path \"$ANNO_PATH\" \
    --output \"$OUTPUT\" \
    --cache_dir \"$CACHE_DIR\" \
    --batch_size \"$BATCH_SIZE\" \
    --max_frames \"$MAX_FRAMES\" \
    --tensor_parallel_size \"$NUM_GPUS\""

# Add subset flag if set
if [ ! -z "$SUBSET" ]; then
    CMD="$CMD --subset $SUBSET"
fi

# Run evaluation
eval $CMD

echo ""
echo "Evaluation complete! Results saved to $OUTPUT"
