#!/usr/bin/env python3
"""
Fast VideoMME Evaluation with vLLM Batching

Uses vLLM for 10-20x speedup through batched inference.
Processes multiple questions simultaneously for maximum throughput.
"""

import os
import sys
import json
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from orchestrator.qwen_agent_vllm import QwenVideoAgentVLLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# VideoMME categories
VIDEO_TYPES = ["short", "medium", "long"]
CATEGORIES = [
    "Knowledge", "Film & Television", "Sports Competition", 
    "Artistic Performance", "Life Record", "Multilingual"
]


def extract_answer(response: str) -> str:
    """Extract answer letter (A, B, C, or D) from model response."""
    response = response.strip()
    
    # Remove common prefixes
    prefixes = [
        "The best answer is", "The correct answer is", "The answer is",
        "The answer", "The best option is", "The correct option is",
    ]
    for prefix in prefixes:
        response = response.replace(prefix, "")
    
    # Find first occurrence of A, B, C, or D
    match = re.search(r'[ABCD]', response)
    if match is None:
        return ""
    return match[0]


def load_videomme_annotations(anno_path: str) -> List[Dict]:
    """Load VideoMME annotations from parquet file."""
    logger.info(f"Loading annotations from {anno_path}")
    
    # Load parquet file
    df = pd.read_parquet(anno_path)
    
    # Convert to list of dicts
    annotations = []
    for _, row in df.iterrows():
        doc = row.to_dict()
        
        # Convert numpy arrays to lists
        for key, value in doc.items():
            if hasattr(value, 'tolist'):
                doc[key] = value.tolist()
        
        annotations.append(doc)
    
    logger.info(f"Loaded {len(annotations)} annotations")
    return annotations


def find_video_path(video_id: str, data_dir: str) -> str:
    """Find video file path."""
    # Try different extensions
    for ext in ['.mp4', '.MP4', '.mkv', '.avi']:
        video_path = Path(data_dir) / f"{video_id}{ext}"
        if video_path.exists():
            return str(video_path)
    
    raise FileNotFoundError(f"Video not found: {video_id}")


def evaluate_videomme(
    data_dir: str,
    anno_path: str,
    output_file: str,
    cache_dir: str = "./cache/videomme_eval_vllm",
    batch_size: int = 16,
    max_samples: int = None,
    max_frames: int = 6,
    duration_filter: str = None,
    checkpoint_every: int = 100,
    subset_fraction: float = None,
    tensor_parallel_size: int = 1
):
    """
    Evaluate with vLLM batched inference.
    
    Args:
        data_dir: Directory containing video files
        anno_path: Path to annotations parquet file
        output_file: Path to save results
        cache_dir: Cache directory
        batch_size: Number of questions to process in parallel
        max_samples: Maximum samples (for testing)
        max_frames: Maximum frames per video
        duration_filter: Filter by duration (short/medium/long)
        checkpoint_every: Save checkpoint every N samples
        subset_fraction: Fraction of dataset to use (e.g., 0.33 for 1/3), sampled proportionally from each duration category
        tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
    """
    # Load annotations
    annotations = load_videomme_annotations(anno_path)
    
    # Apply subset sampling before duration filter (to maintain proportions)
    if subset_fraction is not None:
        # Group by duration category
        by_duration = defaultdict(list)
        for anno in annotations:
            duration = anno.get('duration', 'unknown')
            by_duration[duration].append(anno)
        
        # Sample from each duration category
        sampled_annotations = []
        for duration, items in by_duration.items():
            n_sample = int(len(items) * subset_fraction)
            sampled = items[:n_sample]  # Take first N items
            sampled_annotations.extend(sampled)
            logger.info(f"Sampled {n_sample}/{len(items)} from {duration} duration category")
        
        annotations = sampled_annotations
        logger.info(f"Total after subset sampling: {len(annotations)} samples")
    
    # Filter by duration
    if duration_filter:
        annotations = [a for a in annotations if a.get('duration') == duration_filter]
        logger.info(f"Filtered to {len(annotations)} {duration_filter} videos")
    
    # Limit for testing
    if max_samples:
        annotations = annotations[:max_samples]
        logger.info(f"Limited to {max_samples} samples")
    
    logger.info(f"Total samples to evaluate: {len(annotations)}")
    
    # Initialize vLLM agent
    logger.info("Initializing vLLM agent (this may take a few minutes)...")
    logger.info(f"Using tensor parallelism across {tensor_parallel_size} GPU(s)")
    agent = QwenVideoAgentVLLM(
        model_name="Qwen/Qwen3-VL-8B-Instruct",  # Use Qwen3-VL with vLLM >= 0.11.0
        cache_dir=cache_dir,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.85,  # Can use more with tensor parallelism
    )
    
    # Process in batches
    all_results = []
    
    for i in tqdm(range(0, len(annotations), batch_size), desc="Processing batches"):
        batch = annotations[i:i+batch_size]
        
        # Prepare batch questions
        questions = []
        for doc in batch:
            video_id = doc.get('videoID', doc.get('video_id', ''))
            
            try:
                video_path = find_video_path(video_id, data_dir)
                
                questions.append({
                    'video_path': video_path,
                    'video_id': video_id,
                    'question_id': doc.get('question_id'),
                    'question': doc['question'],
                    'options': doc.get('options', []),
                    'ground_truth': doc.get('answer', ''),
                    'duration': doc.get('duration', ''),
                    'domain': doc.get('domain', ''),
                    'sub_category': doc.get('sub_category', ''),
                    'task_type': doc.get('task_type', '')
                })
            except FileNotFoundError as e:
                logger.warning(f"Skipping {video_id}: {e}")
                continue
        
        if not questions:
            continue
        
        # Batch inference!
        try:
            batch_results = agent.answer_questions_batch(
                questions, 
                max_frames=max_frames
            )
            
            # Add ground truth and compute correctness
            for result, q in zip(batch_results, questions):
                result['ground_truth'] = q['ground_truth']
                result['duration'] = q['duration']
                result['domain'] = q['domain']
                result['sub_category'] = q['sub_category']
                result['task_type'] = q['task_type']
                
                # Extract predicted answer
                pred = extract_answer(result['answer'])
                result['predicted_answer'] = pred
                result['correct'] = (pred == q['ground_truth'])
            
            all_results.extend(batch_results)
            
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size}: {e}")
            continue
        
        # Checkpoint
        if (i + batch_size) % checkpoint_every == 0:
            checkpoint_path = output_file.replace('.json', f'_checkpoint_{i+batch_size}.json')
            with open(checkpoint_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Calculate metrics
    metrics = calculate_metrics(all_results)
    
    # Save final results
    final_output = {
        'results': all_results,
        'metrics': metrics,
        'config': {
            'batch_size': batch_size,
            'max_frames': max_frames,
            'duration_filter': duration_filter,
            'subset_fraction': subset_fraction,
            'tensor_parallel_size': tensor_parallel_size,
            'total_samples': len(all_results)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")
    print_metrics(metrics)
    
    return metrics


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics."""
    metrics = {
        'overall': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_duration': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_category': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_task': defaultdict(lambda: {'correct': 0, 'total': 0})
    }
    
    for r in results:
        correct = r.get('correct', False)
        
        metrics['overall']['all']['total'] += 1
        if correct:
            metrics['overall']['all']['correct'] += 1
        
        duration = r.get('duration', 'unknown')
        metrics['by_duration'][duration]['total'] += 1
        if correct:
            metrics['by_duration'][duration]['correct'] += 1
        
        category = r.get('domain', 'unknown')
        metrics['by_category'][category]['total'] += 1
        if correct:
            metrics['by_category'][category]['correct'] += 1
        
        task = r.get('task_type', 'unknown')
        metrics['by_task'][task]['total'] += 1
        if correct:
            metrics['by_task'][task]['correct'] += 1
    
    # Convert to regular dicts and add accuracy
    def add_accuracy(d):
        return {
            k: {
                **v,
                'accuracy': 100.0 * v['correct'] / v['total'] if v['total'] > 0 else 0.0
            }
            for k, v in d.items()
        }
    
    return {
        'overall': add_accuracy(metrics['overall']),
        'by_duration': add_accuracy(metrics['by_duration']),
        'by_category': add_accuracy(metrics['by_category']),
        'by_task': add_accuracy(metrics['by_task'])
    }


def print_metrics(metrics: Dict):
    """Print evaluation metrics."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Overall
    overall = metrics['overall']['all']
    print(f"\nOverall: {overall['accuracy']:.1f}% ({overall['correct']}/{overall['total']})")
    
    # By duration
    print("\nBy Duration:")
    for dur in VIDEO_TYPES:
        if dur in metrics['by_duration']:
            m = metrics['by_duration'][dur]
            print(f"  {dur}: {m['accuracy']:.1f}% ({m['correct']}/{m['total']})")
    
    # By category
    print("\nBy Category:")
    for cat in sorted(metrics['by_category'].keys()):
        m = metrics['by_category'][cat]
        print(f"  {cat}: {m['accuracy']:.1f}% ({m['correct']}/{m['total']})")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Fast VideoMME Evaluation with vLLM")
    parser.add_argument("--data_dir", required=True, help="Video directory")
    parser.add_argument("--anno_path", required=True, help="Annotations parquet file")
    parser.add_argument("--output", default="videomme_results_vllm.json", help="Output file")
    parser.add_argument("--cache_dir", default="./cache/videomme_vllm", help="Cache directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--max_samples", type=int, help="Max samples (for testing)")
    parser.add_argument("--max_frames", type=int, default=6, help="Max frames per video")
    parser.add_argument("--duration", choices=VIDEO_TYPES, help="Filter by duration")
    parser.add_argument("--checkpoint_every", type=int, default=100, help="Checkpoint frequency")
    parser.add_argument("--subset", type=float, help="Use subset of dataset (e.g., 0.33 for 1/3), sampled proportionally from each duration category")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism (e.g., 8 for 8 GPUs)")
    
    args = parser.parse_args()
    
    evaluate_videomme(
        data_dir=args.data_dir,
        anno_path=args.anno_path,
        output_file=args.output,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_frames=args.max_frames,
        duration_filter=args.duration,
        checkpoint_every=args.checkpoint_every,
        subset_fraction=args.subset,
        tensor_parallel_size=args.tensor_parallel_size
    )


if __name__ == "__main__":
    main()
