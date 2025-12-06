#!/usr/bin/env python3
"""
Qwen3-VL Agent with vLLM for Fast Inference

Uses vLLM's offline batched inference for significantly faster processing.
Expected speedup: 10-20x compared to transformers.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
import numpy as np

try:
    from vllm import LLM, SamplingParams
    from vllm.multimodal.utils import fetch_image
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

from tools.sampler import VideoSampler
from tools.embedders import VisualEmbedder
from index.faiss_index import FAISSIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenVideoAgentVLLM:
    """
    Fast video understanding agent using vLLM for batched inference.
    
    Key optimizations:
    1. vLLM's optimized attention with PagedAttention
    2. Batched inference for multiple questions
    3. Shared video preprocessing cache
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        cache_dir: str = "./cache",
        device: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.80,
        max_model_len: int = 8192,
    ):
        """
        Initialize the agent with vLLM.
        
        Args:
            model_name: HuggingFace model ID (Qwen3-VL-8B-Instruct)
            cache_dir: Directory for caching intermediate results
            device: Device to use (auto-detected if None)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum context length
        """
        if not HAS_VLLM:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            )
        
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing QwenVideoAgentVLLM on {self.device}")
        logger.info(f"Model: {model_name}")
        
        # Initialize tools (lazy loading)
        self.sampler = None
        self.embedder = None
        self.index = None
        
        # Cache for video preprocessing (video_id -> frames)
        self.video_cache = {}
        
        # Initialize vLLM
        logger.info("Loading vLLM model (this may take a few minutes)...")
        
        # Disable Triton for CUDA 13.0 / V100 compatibility
        import os
        os.environ['VLLM_USE_TRITON_FLASH_ATTN'] = '0'
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": 20},  # Allow up to 20 images per prompt
            trust_remote_code=True,
            disable_custom_all_reduce=True  # Disable custom kernels for compatibility
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop_token_ids=[]
        )
        
        logger.info("vLLM model loaded successfully")
    
    def _init_tools(self):
        """Lazy initialize perception tools."""
        if self.sampler is None:
            self.sampler = VideoSampler(
                cache_dir=str(self.cache_dir),
                deduplicate=True
            )
        
        if self.embedder is None:
            self.embedder = VisualEmbedder(
                model_name="facebook/dinov2-large",
                cache_dir=str(self.cache_dir / "embeddings"),
                use_cache=True
            )
    
    def preprocess_video(
        self,
        video_path: str,
        video_id: str,
        num_frames_per_shot: int = 1,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Extract frames and build search index.
        Results are cached by video_id.
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for caching
            num_frames_per_shot: Keyframes to extract per shot
            force: Force reprocessing even if cached
            
        Returns:
            Dict with frames and metadata
        """
        # Check cache
        if not force and video_id in self.video_cache:
            logger.info(f"Using cached preprocessing for {video_id}")
            return self.video_cache[video_id]
        
        self._init_tools()
        
        logger.info(f"Preprocessing video: {video_id}")
        
        # 1. Sample keyframes
        frames = self.sampler.sample_keyframes(
            video_path,
            num_frames_per_shot=num_frames_per_shot
        )
        
        # Store result
        result = {
            "frames": frames,
            "num_frames": len(frames),
            "video_id": video_id
        }
        
        self.video_cache[video_id] = result
        return result
    
    def answer_questions_batch(
        self,
        questions: List[Dict[str, Any]],
        max_frames: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions in a single batch (FAST!).
        
        Args:
            questions: List of dicts with keys:
                - video_path: Path to video
                - video_id: Unique ID for caching
                - question: Question text
                - options: List of answer options (optional)
            max_frames: Maximum frames to use per question
            
        Returns:
            List of answer dicts
        """
        # Preprocess all unique videos
        unique_videos = {}
        for q in questions:
            vid = q['video_id']
            if vid not in unique_videos and vid not in self.video_cache:
                unique_videos[vid] = q['video_path']
        
        logger.info(f"Preprocessing {len(unique_videos)} unique videos...")
        for video_id, video_path in unique_videos.items():
            self.preprocess_video(video_path, video_id)
        
        # Build prompts for all questions
        prompts = []
        metadata = []
        
        for q in questions:
            video_data = self.video_cache[q['video_id']]
            frames = video_data['frames']
            
            # Select frames
            step = max(1, len(frames) // max_frames)
            selected_frames = frames[::step][:max_frames]
            
            # Build multimodal prompt
            content = []
            for frame in selected_frames:
                content.append({"type": "image", "image": frame['path']})
            
            # Add question text
            question_text = q['question']
            if 'options' in q and q['options']:
                question_text += "\nOptions:\n"
                for i, opt in enumerate(q['options']):
                    question_text += f"{chr(65+i)}. {opt}\n"
                question_text += "\nSelect the best answer (A, B, C, or D)."
            
            content.append({"type": "text", "text": question_text})
            
            # Format as vLLM expects
            prompt = {
                "prompt": self._format_prompt_vllm(content),
                "multi_modal_data": {
                    "image": [frame['path'] for frame in selected_frames]
                }
            }
            
            prompts.append(prompt)
            metadata.append({
                "question_id": q.get('question_id'),
                "video_id": q['video_id'],
                "num_frames": len(selected_frames)
            })
        
        # Batch inference (THIS IS THE MAGIC!)
        logger.info(f"Running batched inference on {len(prompts)} questions...")
        outputs = self.llm.generate(
            prompts,
            sampling_params=self.sampling_params
        )
        
        # Parse results
        results = []
        for output, meta, q in zip(outputs, metadata, questions):
            generated_text = output.outputs[0].text
            
            result = {
                "question_id": meta['question_id'],
                "video_id": meta['video_id'],
                "question": q['question'],
                "answer": generated_text,
                "num_frames_used": meta['num_frames']
            }
            
            # Extract answer letter if MCQ
            if 'options' in q and q['options']:
                import re
                match = re.search(r'[ABCD]', generated_text)
                if match:
                    result["selected_option"] = match[0]
                    result["selected_option_idx"] = ord(match[0]) - ord('A')
            
            results.append(result)
        
        return results
    
    def _format_prompt_vllm(self, content: List[Dict]) -> str:
        """Format multimodal content for vLLM."""
        # vLLM expects images as <image> tokens
        parts = []
        for item in content:
            if item["type"] == "image":
                parts.append("<image>")
            elif item["type"] == "text":
                parts.append(item["text"])
        return "\n".join(parts)
    
    def answer_question(
        self,
        video_path: str,
        video_id: str,
        question: str,
        options: Optional[List[str]] = None,
        max_frames: int = 6
    ) -> Dict[str, Any]:
        """
        Answer a single question (convenience wrapper).
        
        For best performance, use answer_questions_batch() instead.
        """
        questions = [{
            "video_path": video_path,
            "video_id": video_id,
            "question": question,
            "options": options
        }]
        
        results = self.answer_questions_batch(questions, max_frames=max_frames)
        return results[0]


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python qwen_agent_vllm.py <video_path> <question>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    question = sys.argv[2]
    
    # Initialize agent
    agent = QwenVideoAgentVLLM(cache_dir="./cache/agent_vllm")
    
    # Answer question
    result = agent.answer_question(
        video_path=video_path,
        video_id=Path(video_path).stem,
        question=question,
        max_frames=4
    )
    
    print("\n" + "="*80)
    print("AGENT RESPONSE")
    print("="*80)
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print("="*80)


if __name__ == "__main__":
    main()
