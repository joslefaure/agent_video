#!/bin/bash
# Multi-GPU VideoMME Evaluation Orchestrator
# This script splits the dataset and launches independent workers on each GPU

set -e

# Configuration
VIDEO_DIR="/work/u1399652/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/video"
ANNO_FILE="/work/u1399652/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/test-00000-of-00001.parquet"
OUTPUT_DIR="./results/videomme_full_multigpu"
CACHE_DIR="./cache/videomme_eval_multigpu"
MAX_FRAMES=8
NUM_GPUS=8

# Create output and cache directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

echo "================================================================================"
echo "VideoMME Multi-GPU Evaluation"
echo "================================================================================"
echo "Video directory: $VIDEO_DIR"
echo "Annotations:     $ANNO_FILE"
echo "Output:          $OUTPUT_DIR"
echo "Cache:           $CACHE_DIR"
echo "GPUs:            $NUM_GPUS"
echo "Max frames:      $MAX_FRAMES"
echo "================================================================================"

# Step 1: Split annotations across GPUs
echo ""
echo "Step 1: Splitting dataset across $NUM_GPUS GPUs..."
python - <<EOF
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Load annotations
anno_path = "$ANNO_FILE"
if anno_path.endswith('.parquet'):
    df = pd.read_parquet(anno_path)
    annotations = df.to_dict('records')
else:
    with open(anno_path, 'r') as f:
        annotations = json.load(f)

# Split across GPUs
num_gpus = $NUM_GPUS
chunk_size = len(annotations) // num_gpus

for gpu_id in range(num_gpus):
    start_idx = gpu_id * chunk_size
    end_idx = start_idx + chunk_size if gpu_id < num_gpus - 1 else len(annotations)
    chunk = annotations[start_idx:end_idx]
    
    # Save chunk with custom encoder
    output_file = f"$CACHE_DIR/annotations_gpu{gpu_id}.json"
    with open(output_file, 'w') as f:
        json.dump(chunk, f, cls=NumpyEncoder)
    
    print(f"GPU {gpu_id}: {len(chunk)} samples -> {output_file}")

print(f"Total: {len(annotations)} samples")
EOF

echo ""
echo "Step 2: Launching workers on each GPU..."

# Launch workers in background
PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Starting worker on GPU $GPU_ID..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python src/eval/evaluate_videomme_single_gpu.py \
        --gpu_id $GPU_ID \
        --annotations_file "$CACHE_DIR/annotations_gpu${GPU_ID}.json" \
        --data_dir "$VIDEO_DIR" \
        --cache_dir "$CACHE_DIR/gpu${GPU_ID}" \
        --max_frames $MAX_FRAMES \
        --output_file "$OUTPUT_DIR/results_gpu${GPU_ID}.json" \
        > "$OUTPUT_DIR/gpu${GPU_ID}.log" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    echo "  Worker PID: $PID, Log: $OUTPUT_DIR/gpu${GPU_ID}.log"
    
    # Stagger starts to avoid simultaneous model loading
    if [ $GPU_ID -lt $((NUM_GPUS - 1)) ]; then
        echo "  Waiting 30 seconds before starting next GPU..."
        sleep 30
    fi
done

echo ""
echo "All workers launched! PIDs: ${PIDS[@]}"
echo "Monitoring logs in $OUTPUT_DIR/*.log"
echo ""
echo "Waiting for all workers to complete..."

# Wait for all workers
for PID in "${PIDS[@]}"; do
    wait $PID
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "WARNING: Worker PID $PID exited with code $EXIT_CODE"
    fi
done

echo ""
echo "Step 3: Merging results..."

# Merge results
python - <<EOF
import json
from pathlib import Path

all_results = []
correct = 0
total = 0

for gpu_id in range($NUM_GPUS):
    result_file = f"$OUTPUT_DIR/results_gpu{gpu_id}.json"
    if Path(result_file).exists():
        with open(result_file, 'r') as f:
            results = json.load(f)
            all_results.extend(results)
            correct += sum(1 for r in results if r.get('correct', False))
            total += len(results)
        print(f"GPU {gpu_id}: {len(results)} results")
    else:
        print(f"WARNING: GPU {gpu_id} results not found!")

# Compute metrics (simplified)
accuracy = 100 * correct / total if total > 0 else 0

output = {
    "overall_accuracy": accuracy,
    "total_questions": total,
    "correct_answers": correct,
    "results": all_results
}

# Save merged results
with open("$OUTPUT_DIR/results.json", 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*80}")
print(f"FINAL RESULTS")
print(f"{'='*80}")
print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
print(f"Results saved to: $OUTPUT_DIR/results.json")
print(f"{'='*80}")
EOF

echo ""
echo "================================================================================"
echo "Evaluation Complete!"
echo "Results: $OUTPUT_DIR/results.json"
echo "Logs: $OUTPUT_DIR/gpu*.log"
echo "================================================================================"
