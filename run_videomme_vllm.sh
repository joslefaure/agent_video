#!/bin/bash
# Fast VideoMME Evaluation with vLLM
# Expected: ~20 seconds per sample instead of 360 seconds (18x speedup!)

DATA_DIR="/home/phd/12/josmyfaure/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/data"
ANNO_PATH="/home/phd/12/josmyfaure/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/test-00000-of-00001.parquet"
OUTPUT="videomme_results_vllm_full.json"
CACHE_DIR="./cache/videomme_vllm"

# Use batch size 16 for maximum throughput
BATCH_SIZE=16
MAX_FRAMES=6

echo "Starting Fast VideoMME Evaluation with vLLM"
echo "==========================================="
echo "Batch size: $BATCH_SIZE"
echo "Max frames: $MAX_FRAMES"
echo "Expected time: ~9 hours (vs 270 hours with transformers)"
echo ""

python src/eval/evaluate_videomme_vllm.py \
    --data_dir "$DATA_DIR" \
    --anno_path "$ANNO_PATH" \
    --output "$OUTPUT" \
    --cache_dir "$CACHE_DIR" \
    --batch_size "$BATCH_SIZE" \
    --max_frames "$MAX_FRAMES" \
    --checkpoint_every 50

echo ""
echo "Evaluation complete! Results saved to $OUTPUT"
