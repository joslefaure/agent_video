#!/usr/bin/env python3
"""
VideoMME Evaluation Script for Qwen3-VL Agent

Based on LMMS-Eval VideoMME evaluation:
https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/videomme
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
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from orchestrator.qwen_agent import QwenVideoAgent

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
SUB_CATEGORIES = [
    "Humanity & History", "Literature & Art", "Biology & Medicine",
    "Finance & Commerce", "Astronomy", "Geography", "Law", "Life Tip",
    "Technology", "Animation", "Movie & TV Show", "Documentary",
    "News Report", "Esports", "Basketball", "Football", "Athletics",
    "Other Sports", "Stage Play", "Magic Show", "Variety Show",
    "Acrobatics", "Handicraft", "Food", "Fashion", "Daily Life",
    "Travel", "Pet & Animal", "Exercise", "Multilingual"
]
TASK_CATEGORIES = [
    "Temporal Perception", "Spatial Perception", "Attribute Perception",
    "Action Recognition", "Object Recognition", "OCR Problems",
    "Counting Problem", "Temporal Reasoning", "Spatial Reasoning",
    "Action Reasoning", "Object Reasoning", "Information Synopsis"
]


def extract_answer(response: str) -> str:
    """
    Extract answer letter (A, B, C, or D) from model response.
    
    Args:
        response: Model's text response
        
    Returns:
        Single letter answer or empty string if not found
    """
    response = response.strip()
    
    # Remove common prefixes
    prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for prefix in prefixes:
        response = response.replace(prefix, "")
    
    # If response is too long and has no letter, return empty
    if len(response.split()) > 10 and not re.search("[ABCD]", response):
        return ""
    
    # Find first occurrence of A, B, C, or D
    match = re.search(r"[ABCD]", response)
    if match is None:
        return ""
    return match[0]


def load_videomme_annotations(anno_path: str) -> List[Dict]:
    """Load VideoMME annotations from JSON or Parquet file."""
    logger.info(f"Loading annotations from {anno_path}")
    
    anno_path = Path(anno_path)
    
    # Handle parquet files
    if anno_path.suffix == '.parquet':
        try:
            import pandas as pd
            import numpy as np
            df = pd.read_parquet(anno_path)
            
            # Convert to dict and handle numpy arrays
            data = []
            for record in df.to_dict('records'):
                # Convert numpy arrays and types to native Python types
                cleaned = {}
                for key, value in record.items():
                    if isinstance(value, np.ndarray):
                        cleaned[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        cleaned[key] = value.item()
                    else:
                        cleaned[key] = value
                data.append(cleaned)
            
            logger.info(f"Loaded {len(data)} annotations from parquet")
            return data
        except ImportError:
            logger.error("pandas required for parquet files. Install: pip install pandas pyarrow")
            raise
    
    # Handle JSON files
    with open(anno_path, 'r') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, dict):
        # Convert dict to list if needed
        annotations = []
        for video_id, questions in data.items():
            if isinstance(questions, list):
                for q in questions:
                    q['videoID'] = video_id
                    annotations.append(q)
            else:
                questions['videoID'] = video_id
                annotations.append(questions)
    else:
        annotations = data
    
    logger.info(f"Loaded {len(annotations)} questions")
    return annotations


def find_video_path(video_id: str, data_dir: str) -> str:
    """Find video file with various possible extensions."""
    data_path = Path(data_dir)
    
    # Try common extensions
    for ext in ['.mp4', '.MP4', '.mkv', '.avi']:
        video_path = data_path / f"{video_id}{ext}"
        if video_path.exists():
            return str(video_path)
    
    # Try in subdirectories
    for video_file in data_path.rglob(f"{video_id}.*"):
        if video_file.suffix.lower() in ['.mp4', '.mkv', '.avi']:
            return str(video_file)
    
    return None


def format_question_prompt(doc: Dict, include_subtitles: bool = False) -> str:
    """
    Format question with options for the model.
    
    Args:
        doc: Annotation document
        include_subtitles: Whether to include subtitle information
        
    Returns:
        Formatted prompt string
    """
    question = doc['question']
    options = doc['options']
    
    # Format options as A, B, C, D
    formatted_options = "\n".join([
        f"{chr(65+i)}. {opt}" 
        for i, opt in enumerate(options)
    ])
    
    prompt = (
        f"Question: {question}\n"
        f"Options:\n{formatted_options}\n\n"
        f"Select the best answer based on the video. "
        f"Respond with only the letter (A, B, C, or D) of the correct option."
    )
    
    return prompt


def evaluate_videomme(
    data_dir: str,
    anno_path: str,
    output_file: str,
    cache_dir: str = "./cache/videomme_eval",
    max_samples: int = None,
    max_frames: int = 8,
    duration_filter: str = None
):
    """
    Evaluate Qwen3-VL agent on VideoMME benchmark.
    
    Args:
        data_dir: Directory containing video files
        anno_path: Path to annotations JSON file
        output_file: Path to save results
        cache_dir: Cache directory for preprocessing
        max_samples: Maximum number of samples to evaluate (for testing)
        max_frames: Maximum frames to use per video
        duration_filter: Filter by duration (short/medium/long)
    """
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
        logger.info(f"Limited to {max_samples} samples for testing")
    
    # Initialize agent
    logger.info("Initializing Qwen3-VL agent...")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    agent = QwenVideoAgent(
        cache_dir=cache_dir,
        load_model=True
    )
    
    # Process each question
    results = []
    correct = 0
    total = 0
    
    for doc in tqdm(annotations, desc="Evaluating VideoMME"):
        video_id = doc.get('videoID', doc.get('video_id', ''))
        question_id = doc.get('question_id', len(results))
        
        # Find video file
        video_path = find_video_path(video_id, data_dir)
        if video_path is None:
            logger.warning(f"Video not found: {video_id}")
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
            if is_correct:
                correct += 1
            total += 1
            
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
            
            # Log progress
            if total % 10 == 0:
                logger.info(f"Progress: {total} questions, Accuracy: {100*correct/total:.1f}%")
            
        except Exception as e:
            logger.error(f"Error processing {video_id}, question {question_id}: {e}")
            continue
    
    # Compute detailed metrics
    metrics = compute_metrics(results)
    
    # Save results
    output = {
        "overall_accuracy": 100 * correct / total if total > 0 else 0,
        "total_questions": total,
        "correct_answers": correct,
        "metrics": metrics,
        "results": results
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")
    logger.info(f"Overall Accuracy: {output['overall_accuracy']:.1f}%")
    
    # Print detailed metrics
    print_metrics(metrics)
    
    return output


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute detailed metrics by category."""
    metrics = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for result in results:
        # Overall
        metrics["overall"]["total"] += 1
        if result["correct"]:
            metrics["overall"]["correct"] += 1
        
        # By duration
        duration = result.get("duration", "unknown")
        metrics[f"duration_{duration}"]["total"] += 1
        if result["correct"]:
            metrics[f"duration_{duration}"]["correct"] += 1
        
        # By category
        category = result.get("category", "unknown")
        metrics[f"category_{category}"]["total"] += 1
        if result["correct"]:
            metrics[f"category_{category}"]["correct"] += 1
        
        # By sub-category
        sub_cat = result.get("sub_category", "unknown")
        metrics[f"sub_category_{sub_cat}"]["total"] += 1
        if result["correct"]:
            metrics[f"sub_category_{sub_cat}"]["correct"] += 1
        
        # By task category
        task = result.get("task_category", "unknown")
        metrics[f"task_{task}"]["total"] += 1
        if result["correct"]:
            metrics[f"task_{task}"]["correct"] += 1
    
    # Convert to percentages
    metrics_pct = {}
    for key, value in metrics.items():
        if value["total"] > 0:
            metrics_pct[key] = {
                "accuracy": 100 * value["correct"] / value["total"],
                "correct": value["correct"],
                "total": value["total"]
            }
    
    return metrics_pct


def print_metrics(metrics: Dict):
    """Print metrics in a readable format."""
    print("\n" + "="*80)
    print("DETAILED METRICS")
    print("="*80)
    
    # Overall
    if "overall" in metrics:
        m = metrics["overall"]
        print(f"\nOverall: {m['accuracy']:.1f}% ({m['correct']}/{m['total']})")
    
    # By duration
    print("\nBy Duration:")
    for duration in VIDEO_TYPES:
        key = f"duration_{duration}"
        if key in metrics:
            m = metrics[key]
            print(f"  {duration}: {m['accuracy']:.1f}% ({m['correct']}/{m['total']})")
    
    # By category
    print("\nBy Category:")
    for cat in CATEGORIES:
        key = f"category_{cat}"
        if key in metrics:
            m = metrics[key]
            print(f"  {cat}: {m['accuracy']:.1f}% ({m['correct']}/{m['total']})")
    
    # By task
    print("\nBy Task:")
    for task in TASK_CATEGORIES:
        key = f"task_{task}"
        if key in metrics:
            m = metrics[key]
            print(f"  {task}: {m['accuracy']:.1f}% ({m['correct']}/{m['total']})")
    
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL agent on VideoMME benchmark"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing video files"
    )
    parser.add_argument(
        "--anno_path",
        type=str,
        required=True,
        help="Path to VideoMME annotations JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/videomme_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache/videomme_eval",
        help="Cache directory for preprocessing"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=8,
        help="Maximum frames per video to use"
    )
    parser.add_argument(
        "--duration",
        type=str,
        choices=["short", "medium", "long"],
        default=None,
        help="Filter by video duration"
    )
    
    args = parser.parse_args()
    
    evaluate_videomme(
        data_dir=args.data_dir,
        anno_path=args.anno_path,
        output_file=args.output,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
        max_frames=args.max_frames,
        duration_filter=args.duration
    )
