#!/bin/bash
set -e

# Ablation study script
# Runs the pipeline with different configurations

DATA_DIR=${1:-"./sample_data"}
ANNO_DIR=${2:-"./sample_data"}

echo "Starting Ablation Study..."

# 1. Baseline
echo "Running Baseline..."
./run.sh --task videomme --data_dir "$DATA_DIR" --anno_dir "$ANNO_DIR" --output "results_baseline.json"

# 2. No OCR (Simulated by not using OCR tool in agent - requires code change or flag)
# For now, we just show how one would structure this. 
# Ideally, pass flags to run.sh -> evaluate_videomme.py -> VLLMAgent

echo "Ablation study complete. Results saved."
