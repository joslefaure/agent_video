#!/usr/bin/env python3
"""
Multi-GPU VideoMME Evaluation Script for Qwen3-VL Agent

This script distributes evaluation across multiple GPUs for faster processing.
Each GPU processes a subset of the data independently.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import torch
import torch.multiprocessing as mp
from functools import partial

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import after path setup
from eval.evaluate_videomme import (
    load_videomme_annotations,
    find_video_path,
    format_question_prompt,
    extract_answer,
    compute_metrics,
    print_metrics
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_subset_on_gpu(
    gpu_id: int,
    annotations: List[Dict],
    data_dir: str,
    cache_dir: str,
    max_frames: int,
    results_queue: mp.Queue
):
    """
    Process a subset of annotations on a specific GPU.
    
    Args:
        gpu_id: GPU device ID to use
        annotations: List of annotations to process
        data_dir: Directory containing videos
        cache_dir: Cache directory
        max_frames: Max frames per video
        results_queue: Queue to put results in
    """
    # CRITICAL: Set environment variable BEFORE importing torch/transformers
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Import here AFTER setting CUDA_VISIBLE_DEVICES
    import torch
    from orchestrator.qwen_agent import QwenVideoAgent
    from tqdm import tqdm
    
    # Verify only one GPU is visible
    assert torch.cuda.device_count() == 1, f"Expected 1 GPU, got {torch.cuda.device_count()}"
    torch.cuda.set_device(0)  # Now device 0 is our assigned GPU
    
    logger.info(f"GPU {gpu_id}: Initializing agent for {len(annotations)} samples")
    
    # Initialize agent on this GPU - it will see only one GPU (index 0)
    agent = QwenVideoAgent(
        cache_dir=f"{cache_dir}/gpu{gpu_id}",
        load_model=True,
        device="cuda"
    )
    
    results = []
    
    # Process assigned annotations
    for doc in tqdm(annotations, desc=f"GPU {gpu_id}", position=gpu_id):
        video_id = doc.get('videoID', doc.get('video_id', ''))
        question_id = doc.get('question_id', len(results))
        
        # Find video file
        video_path = find_video_path(video_id, data_dir)
        if video_path is None:
            logger.warning(f"GPU {gpu_id}: Video not found: {video_id}")
            continue
        
        # Format question
        question = format_question_prompt(doc)
        options = doc['options']
        gt_answer = doc['answer']
        
        try:
            # Get model prediction
            response = agent.answer_question(
                video_path,
                question,
                options=options,
                max_frames=max_frames
            )
            
            # Extract answer
            pred_text = response['answer']
            pred_answer = extract_answer(pred_text)
            
            # Check correctness
            is_correct = pred_answer == gt_answer
            
            # Store result
            result = {
                "question_id": question_id,
                "video_id": video_id,
                "duration": doc.get('duration', ''),
                "category": doc.get('domain', ''),
                "sub_category": doc.get('sub_category', ''),
                "task_category": doc.get('task_type', ''),
                "question": doc['question'],
                "options": options,
                "ground_truth": gt_answer,
                "prediction": pred_answer,
                "prediction_text": pred_text,
                "correct": is_correct,
                "evidence_frames": response.get('evidence_frames', [])
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"GPU {gpu_id}: Error processing {video_id}, question {question_id}: {str(e)}")
            # Clear CUDA cache on error
            import torch
            torch.cuda.empty_cache()
            continue
        
        # Periodically clear cache to prevent fragmentation
        if len(results) % 10 == 0:
            import torch
            torch.cuda.empty_cache()
    
    # Put results in queue
    results_queue.put(results)
    logger.info(f"GPU {gpu_id}: Completed {len(results)} samples")


def evaluate_videomme_multigpu(
    data_dir: str,
    anno_path: str,
    output_file: str,
    cache_dir: str = "./cache/videomme_eval",
    max_samples: int = None,
    max_frames: int = 8,
    duration_filter: str = None,
    num_gpus: int = None
):
    """
    Evaluate Qwen3-VL agent on VideoMME using multiple GPUs.
    
    Args:
        data_dir: Directory containing video files
        anno_path: Path to annotations file
        output_file: Path to save results
        cache_dir: Cache directory
        max_samples: Maximum samples to evaluate
        max_frames: Max frames per video
        duration_filter: Filter by duration
        num_gpus: Number of GPUs to use (default: all available)
    """
    # Determine number of GPUs
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    logger.info(f"Using {num_gpus} GPUs for evaluation")
    
    # Load annotations
    annotations = load_videomme_annotations(anno_path)
    
    # Filter by duration if specified
    if duration_filter:
        annotations = [
            a for a in annotations 
            if a.get('duration', '') == duration_filter
        ]
        logger.info(f"Filtered to {len(annotations)} {duration_filter} videos")
    
    # Limit samples for testing
    if max_samples:
        annotations = annotations[:max_samples]
        logger.info(f"Limited to {max_samples} samples")
    
    # Split annotations across GPUs
    chunk_size = len(annotations) // num_gpus
    annotation_chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_gpus - 1 else len(annotations)
        annotation_chunks.append(annotations[start_idx:end_idx])
        logger.info(f"GPU {i}: Assigned {len(annotation_chunks[i])} samples")
    
    # Create multiprocessing context
    mp.set_start_method('spawn', force=True)
    results_queue = mp.Queue()
    
    # Launch processes for each GPU with staggered starts
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=process_subset_on_gpu,
            args=(
                gpu_id,
                annotation_chunks[gpu_id],
                data_dir,
                cache_dir,
                max_frames,
                results_queue
            )
        )
        p.start()
        processes.append(p)
        
        # Stagger process starts to avoid simultaneous model loading
        # Give each process 30 seconds to load the model before starting the next
        if gpu_id < num_gpus - 1:  # Don't wait after the last one
            logger.info(f"Waiting 30s before starting GPU {gpu_id + 1}...")
            import time
            time.sleep(30)
    
    # Collect results from all processes
    all_results = []
    for _ in range(num_gpus):
        results = results_queue.get()
        all_results.extend(results)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    logger.info(f"Collected {len(all_results)} total results")
    
    # Compute metrics
    metrics = compute_metrics(all_results)
    
    # Calculate overall accuracy
    correct = sum(1 for r in all_results if r.get('correct', False))
    total = len(all_results)
    overall_accuracy = 100 * correct / total if total > 0 else 0
    
    # Save results
    output = {
        "overall_accuracy": overall_accuracy,
        "total_questions": total,
        "correct_answers": correct,
        "metrics": metrics,
        "results": all_results
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")
    
    logger.info("="*80)
    logger.info("FINAL RESULTS")
    logger.info("="*80)
    logger.info(f"Overall Accuracy: {overall_accuracy:.1f}%")
    logger.info(f"Total: {correct}/{total}")
    logger.info("="*80)
    
    # Print detailed metrics
    print_metrics(metrics)


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU VideoMME Evaluation")
    parser.add_argument("--data_dir", type=str, required=True, help="Video directory")
    parser.add_argument("--anno_path", type=str, required=True, help="Annotations file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--cache_dir", type=str, default="./cache/videomme_eval", help="Cache directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--max_frames", type=int, default=8, help="Max frames per video")
    parser.add_argument("--duration", type=str, default=None, choices=["short", "medium", "long"], help="Filter by duration")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    evaluate_videomme_multigpu(
        data_dir=args.data_dir,
        anno_path=args.anno_path,
        output_file=args.output,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
        max_frames=args.max_frames,
        duration_filter=args.duration,
        num_gpus=args.num_gpus
    )


if __name__ == "__main__":
    main()
