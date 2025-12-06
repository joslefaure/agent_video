#!/usr/bin/env python3
"""
Qwen3-VL Agent for Video Understanding

This agent orchestrates multiple perception tools to answer questions about videos.
It uses Qwen3-VL-8B-Instruct as the VLM backbone with tool-calling capabilities.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np

from tools.sampler import VideoSampler
from tools.embedders import VisualEmbedder
from tools.ocr import OCRExtractor
from index.faiss_index import FAISSIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenVideoAgent:
    """
    Agentic video understanding with Qwen3-VL.
    
    The agent can:
    1. Sample and index video frames
    2. Search for relevant frames using FAISS
    3. Extract text using OCR
    4. Reason about the video to answer questions
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        cache_dir: str = "./cache",
        device: Optional[str] = None,
        load_model: bool = True
    ):
        """
        Initialize the agent.
        
        Args:
            model_name: HuggingFace model ID
            cache_dir: Directory for caching intermediate results
            device: Device to use (cuda/cpu). Auto-detected if None.
            load_model: Whether to load the VLM (set False for tool-only testing)
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing QwenVideoAgent on {self.device}")
        
        # Initialize tools (lazy loading)
        self.sampler = None
        self.embedder = None
        self.ocr = None
        self.index = None
        
        # VLM components
        self.model = None
        self.processor = None
        
        if load_model:
            self._load_model()
    
    def _load_model(self):
        """Load Qwen3-VL model and processor."""
        logger.info(f"Loading VLM: {self.model_name}")
        
        # Check available GPUs
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"Found {num_gpus} GPUs - will use automatic model parallelism")
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,  # Auto-distributes across GPUs
            trust_remote_code=True,
            max_memory={i: "20GB" for i in range(torch.cuda.device_count())} if torch.cuda.is_available() else None  # Limit per-GPU memory
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model.eval()
        logger.info("VLM loaded successfully")
    
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
        
        if self.ocr is None:
            # OCR has issues, skip for now
            pass
    
    def preprocess_video(
        self,
        video_path: str,
        num_frames_per_shot: int = 1,
        build_index: bool = True
    ) -> Dict[str, Any]:
        """
        Extract frames and build search index.
        
        Args:
            video_path: Path to video file
            num_frames_per_shot: Keyframes to extract per shot
            build_index: Whether to build FAISS index
            
        Returns:
            Dict with frames, index, and metadata
        """
        self._init_tools()
        
        logger.info(f"Preprocessing video: {video_path}")
        
        # 1. Sample keyframes
        logger.info("Step 1: Sampling keyframes...")
        frames = self.sampler.sample_keyframes(
            video_path,
            num_frames_per_shot=num_frames_per_shot
        )
        logger.info(f"Extracted {len(frames)} frames")
        
        # 2. Compute embeddings
        logger.info("Step 2: Computing embeddings...")
        frames = self.embedder.embed_frames(
            frames,
            batch_size=32,
            show_progress=True
        )
        logger.info(f"Generated {len(frames)} embeddings")
        
        # 3. Build index
        if build_index:
            logger.info("Step 3: Building FAISS index...")
            embedding_dim = self.embedder.get_embedding_dim()
            self.index = FAISSIndex(
                dimension=embedding_dim,
                index_type="Flat",
                metric="IP"
            )
            self.index.add_from_frames(frames, normalize=True)
            logger.info(f"Index built with {len(self.index)} items")
        
        return {
            "frames": frames,
            "index": self.index,
            "num_frames": len(frames)
        }
    
    def search_frames(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        normalize: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar frames using FAISS.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            normalize: Whether to normalize embeddings
            
        Returns:
            List of frame metadata with similarity scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call preprocess_video first.")
        
        results = self.index.search(query_embedding, k=k, normalize=normalize)
        return results
    
    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0
    ) -> str:
        """
        Generate response from Qwen2.5-VL.
        
        Args:
            messages: Chat messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        if self.model is None:
            raise ValueError("Model not loaded. Set load_model=True or call _load_model()")
        
        # Process vision info (images/videos in messages)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response
    
    def answer_question(
        self,
        video_path: str,
        question: str,
        options: Optional[List[str]] = None,
        use_frames: Optional[List[str]] = None,
        max_frames: int = 8
    ) -> Dict[str, Any]:
        """
        Answer a question about a video.
        
        Args:
            video_path: Path to video file
            question: Question to answer
            options: Multiple choice options (optional)
            use_frames: Specific frame paths to use (optional)
            max_frames: Maximum frames to show to VLM
            
        Returns:
            Dict with answer, reasoning, and evidence frames
        """
        # Preprocess if not already done
        if self.index is None:
            video_data = self.preprocess_video(video_path)
            frames = video_data["frames"]
        else:
            # Assume frames are already loaded
            frames = self.index.metadata if hasattr(self.index, 'metadata') else []
        
        # Select frames to show
        if use_frames:
            selected_frames = [f for f in frames if f['path'] in use_frames]
        else:
            # Simple strategy: uniformly sample frames
            step = max(1, len(frames) // max_frames)
            selected_frames = frames[::step][:max_frames]
        
        logger.info(f"Using {len(selected_frames)} frames for VLM")
        
        # Build message with frames
        frame_contents = []
        for i, frame in enumerate(selected_frames):
            frame_contents.append({
                "type": "image",
                "image": frame['path']
            })
            # Add timestamp context
            frame_contents.append({
                "type": "text",
                "text": f"[Frame {i+1} at {frame['timestamp']:.1f}s]"
            })
        
        # Build prompt
        prompt_text = f"Question: {question}\n"
        if options:
            prompt_text += "Options:\n"
            for i, opt in enumerate(options):
                prompt_text += f"{chr(65+i)}. {opt}\n"
            prompt_text += "\nPlease select the correct answer and explain your reasoning."
        else:
            prompt_text += "\nPlease answer the question based on what you see in the video frames."
        
        messages = [
            {
                "role": "user",
                "content": frame_contents + [{"type": "text", "text": prompt_text}]
            }
        ]
        
        # Generate response
        response = self.generate_response(messages, max_tokens=512)
        
        # Parse response
        result = {
            "question": question,
            "answer": response,
            "evidence_frames": [f['frame_id'] for f in selected_frames],
            "num_frames_used": len(selected_frames)
        }
        
        # Try to extract answer from options if MCQ
        if options:
            response_upper = response.upper()
            for i, opt in enumerate(options):
                letter = chr(65+i)
                if letter in response_upper or opt.lower() in response.lower():
                    result["selected_option"] = letter
                    result["selected_option_idx"] = i
                    result["selected_option_text"] = opt
                    break
        
        return result


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python qwen_agent.py <video_path> <question> [option1] [option2] ...")
        print("\nExample:")
        print('  python qwen_agent.py video.mp4 "What color is the car?" "Red" "Blue" "Green"')
        sys.exit(1)
    
    video_path = sys.argv[1]
    question = sys.argv[2]
    options = sys.argv[3:] if len(sys.argv) > 3 else None
    
    # Initialize agent
    agent = QwenVideoAgent(cache_dir="./cache/agent_test")
    
    # Answer question
    result = agent.answer_question(video_path, question, options)
    
    # Print result
    print("\n" + "="*80)
    print("AGENT RESPONSE")
    print("="*80)
    print(f"\nQuestion: {result['question']}")
    if options:
        print(f"Options: {options}")
    print(f"\nAnswer: {result['answer']}")
    if 'selected_option' in result:
        print(f"Selected: {result['selected_option']}. {result['selected_option_text']}")
    print(f"\nEvidence frames: {len(result['evidence_frames'])} frames")
    print("="*80)


if __name__ == "__main__":
    main()
