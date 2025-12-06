#!/bin/bash
# Run VideoMME evaluation on the full dataset

export CUDA_VISIBLE_DEVICES=0

set -e

# Configuration
VIDEO_DIR="/home/phd/12/josmyfaure/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/data"
ANNO_FILE="/home/phd/12/josmyfaure/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/test-00000-of-00001.parquet"
OUTPUT_DIR="./results/videomme_full"
CACHE_DIR="./cache/videomme_eval"

# Check if files exist
if [ ! -d "$VIDEO_DIR" ]; then
    echo "Error: Video directory not found: $VIDEO_DIR"
    exit 1
fi

if [ ! -f "$ANNO_FILE" ]; then
    echo "Error: Annotation file not found: $ANNO_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "================================================================================"
echo "VideoMME Evaluation - Full Dataset"
echo "================================================================================"
echo "Video directory: $VIDEO_DIR"
echo "Annotations:     $ANNO_FILE"
echo "Output:          $OUTPUT_DIR"
echo "Cache:           $CACHE_DIR"
echo "================================================================================"
echo ""

# Run evaluation
python src/eval/evaluate_videomme.py \
    --data_dir "$VIDEO_DIR" \
    --anno_path "$ANNO_FILE" \
    --output "$OUTPUT_DIR/results.json" \
    --cache_dir "$CACHE_DIR" \
    --max_frames 8

echo ""
echo "================================================================================"
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "================================================================================"
