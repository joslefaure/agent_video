#!/usr/bin/env python3
"""
vLLM-based Qwen3-VL Agent for efficient batch inference

Based on: https://github.com/QwenLM/Qwen3-VL
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch

# Set environment for vLLM
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenVLLMAgent:
    """
    Efficient Qwen3-VL agent using vLLM for batched inference.
    
    Supports:
    - Multi-GPU tensor parallelism
    - Efficient batched inference
    - Video frame sampling via qwen-vl-utils
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        tensor_parallel_size: int = None,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9
    ):
        """
        Initialize vLLM agent.
        
        Args:
            model_name: HuggingFace model ID
            tensor_parallel_size: Number of GPUs for tensor parallelism (None = auto)
            max_model_len: Maximum sequence length
            gpu_memory_utilization: GPU memory utilization (0-1)
        """
        self.model_name = model_name
        
        # Auto-detect tensor parallel size
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()
        
        logger.info(f"Initializing vLLM with {tensor_parallel_size} GPUs")
        logger.info(f"Model: {model_name}")
        
        # Initialize vLLM
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 10, "video": 1}  # Allow up to 10 images or 1 video per prompt
        )
        
        # Initialize processor for message formatting
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            top_p=1.0,
        )
        
        logger.info("vLLM agent initialized successfully")
    
    def prepare_inputs_for_vllm(
        self,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Prepare inputs for vLLM inference.
        
        Args:
            messages: Chat messages in OpenAI format
            
        Returns:
            Dict with prompt and multi_modal_data
        """
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process vision info (requires qwen_vl_utils 0.0.14+)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,  # 16 for Qwen3-VL
            return_video_kwargs=True,
            return_video_metadata=True
        )
        
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        
        return {
            'prompt': text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs
        }
    
    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, Any]]],
        sampling_params: Optional[SamplingParams] = None
    ) -> List[str]:
        """
        Generate responses for a batch of message lists.
        
        Args:
            messages_batch: List of message lists
            sampling_params: Optional custom sampling parameters
            
        Returns:
            List of generated text responses
        """
        if sampling_params is None:
            sampling_params = self.sampling_params
        
        # Prepare inputs for all messages
        inputs = [
            self.prepare_inputs_for_vllm(messages)
            for messages in messages_batch
        ]
        
        # Generate with vLLM (batched)
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        
        # Extract text from outputs
        responses = [output.outputs[0].text for output in outputs]
        
        return responses
    
    def answer_question(
        self,
        video_path: str,
        question: str,
        options: Optional[List[str]] = None,
        max_frames: int = 4
    ) -> Dict[str, Any]:
        """
        Answer a single question about a video.
        
        Args:
            video_path: Path to video file
            question: Question text
            options: Multiple choice options
            max_frames: Maximum frames to sample (for images)
            
        Returns:
            Dict with answer and metadata
        """
        # Build message content
        content = [
            {
                "type": "video",
                "video": video_path,
                "fps": 1.0  # Sample at 1 fps
            },
            {
                "type": "text",
                "text": self._format_question_prompt(question, options)
            }
        ]
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # Generate response
        responses = self.generate_batch([messages])
        response_text = responses[0]
        
        return {
            "question": question,
            "answer": response_text,
            "selected_option": self._extract_answer(response_text) if options else None
        }
    
    def answer_questions_batch(
        self,
        video_paths: List[str],
        questions: List[str],
        options_list: List[Optional[List[str]]],
        max_frames: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions in a batch (efficient).
        
        Args:
            video_paths: List of video paths
            questions: List of questions
            options_list: List of option lists (or None)
            max_frames: Maximum frames per video
            
        Returns:
            List of answer dicts
        """
        # Build messages for all questions
        messages_batch = []
        for video_path, question, options in zip(video_paths, questions, options_list):
            content = [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": 1.0
                },
                {
                    "type": "text",
                    "text": self._format_question_prompt(question, options)
                }
            ]
            
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            messages_batch.append(messages)
        
        # Generate all responses in batch
        responses = self.generate_batch(messages_batch)
        
        # Format results
        results = []
        for i, response_text in enumerate(responses):
            result = {
                "question": questions[i],
                "answer": response_text,
                "selected_option": self._extract_answer(response_text) if options_list[i] else None
            }
            results.append(result)
        
        return results
    
    def _format_question_prompt(
        self,
        question: str,
        options: Optional[List[str]] = None
    ) -> str:
        """Format question with options for the model."""
        prompt = f"Question: {question}\n"
        
        if options:
            prompt += "Options:\n"
            for i, opt in enumerate(options):
                prompt += f"{chr(65+i)}. {opt}\n"
            prompt += "\nPlease select the correct answer (A, B, C, or D) and explain your reasoning."
        else:
            prompt += "\nPlease answer the question based on the video."
        
        return prompt
    
    def _extract_answer(self, response: str) -> str:
        """Extract answer letter from response."""
        import re
        response = response.strip().upper()
        
        # Find first occurrence of A, B, C, or D
        match = re.search(r'[ABCD]', response)
        if match:
            return match[0]
        return ""


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python qwen_vllm_agent.py <video_path> <question> [option1] [option2] ...")
        sys.exit(1)
    
    video_path = sys.argv[1]
    question = sys.argv[2]
    options = sys.argv[3:] if len(sys.argv) > 3 else None
    
    # Initialize agent
    agent = QwenVLLMAgent()
    
    # Answer question
    result = agent.answer_question(video_path, question, options)
    
    print("\n" + "="*80)
    print("VLLM AGENT RESPONSE")
    print("="*80)
    print(f"Question: {question}")
    if options:
        print(f"Options: {options}")
    print(f"\nAnswer: {result['answer']}")
    if result['selected_option']:
        print(f"Selected: {result['selected_option']}")
    print("="*80)


if __name__ == "__main__":
    main()
