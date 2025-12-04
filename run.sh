#!/bin/bash
set -e

# Activate conda environment if needed (optional, assumes user might have it active)
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate agentic_video

# Default arguments
TASK="videomme"
DATA_DIR="./sample_data"
ANNO_DIR="./sample_data" # For smoke test, we might need dummy annotations
OUTPUT="results.json"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --task) TASK="$2"; shift ;;
        --data_dir) DATA_DIR="$2"; shift ;;
        --anno_dir) ANNO_DIR="$2"; shift ;;
        --output) OUTPUT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Running pipeline for task: $TASK"

if [ "$TASK" == "videomme" ]; then
    # Ensure python path
    export PYTHONPATH=$PYTHONPATH:.
    
    python3 src/eval/evaluate_videomme.py \
        --data_dir "$DATA_DIR" \
        --anno_dir "$ANNO_DIR" \
        --output "$OUTPUT"
        
elif [ "$TASK" == "smoke_test" ]; then
    echo "Running smoke tests..."
    export PYTHONPATH=$PYTHONPATH:.
    pytest tests/test_smoke.py
else
    echo "Unknown task: $TASK"
    exit 1
fi
