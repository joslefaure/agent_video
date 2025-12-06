#!/usr/bin/env python3
"""
Single GPU worker for VideoMME evaluation.
This is launched by the multi-GPU orchestrator script.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List

# CRITICAL: Set CUDA device BEFORE any imports
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    print("ERROR: CUDA_VISIBLE_DEVICES not set!")
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Now safe to import torch and other modules
import torch
from tqdm import tqdm

from orchestrator.qwen_agent import QwenVideoAgent
from eval.evaluate_videomme import (
    find_video_path,
    format_question_prompt,
    extract_answer,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--annotations_file", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--output_file", type=str, required=True)
    
    args = parser.parse_args()
    
    # Verify GPU setup
    assert torch.cuda.device_count() == 1, f"Expected 1 GPU, got {torch.cuda.device_count()}"
    logger.info(f"GPU {args.gpu_id}: Starting worker, CUDA device count: {torch.cuda.device_count()}")
    
    # Load annotations
    with open(args.annotations_file, 'r') as f:
        annotations = json.load(f)
    
    logger.info(f"GPU {args.gpu_id}: Processing {len(annotations)} samples")
    
    # Initialize agent
    agent = QwenVideoAgent(
        cache_dir=args.cache_dir,
        load_model=True,
        device="cuda"
    )
    
    logger.info(f"GPU {args.gpu_id}: Agent initialized, starting evaluation")
    
    results = []
    
    # Process each annotation
    for doc in tqdm(annotations, desc=f"GPU {args.gpu_id}", position=args.gpu_id):
        video_id = doc.get('videoID', doc.get('video_id', ''))
        question_id = doc.get('question_id', len(results))
        
        # Find video file
        video_path = find_video_path(video_id, args.data_dir)
        if video_path is None:
            logger.warning(f"GPU {args.gpu_id}: Video not found: {video_id}")
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
                max_frames=args.max_frames
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
            logger.error(f"GPU {args.gpu_id}: Error processing {video_id}, question {question_id}: {str(e)}")
            torch.cuda.empty_cache()
            continue
        
        # Periodically clear cache
        if len(results) % 10 == 0:
            torch.cuda.empty_cache()
    
    # Save results
    logger.info(f"GPU {args.gpu_id}: Completed {len(results)} samples, saving to {args.output_file}")
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"GPU {args.gpu_id}: Done!")


if __name__ == "__main__":
    main()
