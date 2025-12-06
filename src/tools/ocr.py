"""
OCR module using GOT-OCR2.0.

Provides text extraction from images with support for plain text,
formatted text, and fine-grained OCR with bounding boxes.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import os
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
import json
import hashlib

logger = logging.getLogger(__name__)


class OCRExtractor:
    """
    OCR text extraction using GOT-OCR2.0.
    
    Features:
    - Plain text OCR
    - Formatted text OCR (preserves layout)
    - Fine-grained OCR with bounding boxes
    - Multi-crop OCR for large images
    - Batch processing
    - Result caching
    """
    
    def __init__(
        self,
        model_name: str = "stepfun-ai/GOT-OCR2_0",
        device: str = None,
        cache_dir: Optional[str] = "./cache/ocr",
        use_cache: bool = True
    ):
        """
        Initialize OCR extractor.
        
        Args:
            model_name: HuggingFace model ID
            device: Device to run on ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache OCR results
            use_cache: Whether to cache results
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        
        if self.use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initialized OCRExtractor with {model_name} on {self.device}")
    
    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self.model is None:
            try:
                logger.info(f"Loading OCR model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map=self.device,
                    use_safetensors=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                self.model = self.model.eval()
                
                logger.info("OCR model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load OCR model: {e}")
                raise
    
    def _get_cache_key(self, image_path: str, ocr_type: str, **kwargs) -> str:
        """Generate cache key for OCR result."""
        path_str = str(Path(image_path).absolute())
        options = f"{ocr_type}:{json.dumps(kwargs, sort_keys=True)}"
        key_string = f"{self.model_name}:{path_str}:{options}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _load_from_cache(self, image_path: str, ocr_type: str, **kwargs) -> Optional[Dict]:
        """Load OCR result from cache if available."""
        if not self.use_cache or not self.cache_dir:
            return None
        
        cache_key = self._get_cache_key(image_path, ocr_type, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for {image_path}: {e}")
        return None
    
    def _save_to_cache(self, image_path: str, ocr_type: str, result: Dict, **kwargs):
        """Save OCR result to cache."""
        if not self.use_cache or not self.cache_dir:
            return
        
        cache_key = self._get_cache_key(image_path, ocr_type, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to save cache for {image_path}: {e}")
    
    def extract_text(
        self,
        image_path: Union[str, Path],
        ocr_type: str = "ocr",
        ocr_box: Optional[str] = None,
        ocr_color: Optional[str] = None,
        multi_crop: bool = False
    ) -> Dict:
        """
        Extract text from image.
        
        Args:
            image_path: Path to image file
            ocr_type: 'ocr' (plain text) or 'format' (formatted text)
            ocr_box: Bounding box for fine-grained OCR (format: 'x1,y1,x2,y2')
            ocr_color: Color for fine-grained OCR (format: 'red', 'blue', etc.)
            multi_crop: Whether to use multi-crop OCR for large images
            
        Returns:
            Dict with 'text', 'type', 'has_text', and optional 'boxes' keys
        """
        image_path = str(image_path)
        
        # Check cache
        cache_kwargs = {
            'ocr_box': ocr_box,
            'ocr_color': ocr_color,
            'multi_crop': multi_crop
        }
        cached = self._load_from_cache(image_path, ocr_type, **cache_kwargs)
        if cached is not None:
            return cached
        
        # Load model
        self._load_model()
        
        try:
            # Call appropriate method based on parameters
            if multi_crop:
                text = self.model.chat_crop(
                    self.tokenizer,
                    image_path,
                    ocr_type=ocr_type
                )
            else:
                # Build kwargs for chat method
                chat_kwargs = {'ocr_type': ocr_type}
                if ocr_box:
                    chat_kwargs['ocr_box'] = ocr_box
                if ocr_color:
                    chat_kwargs['ocr_color'] = ocr_color
                
                text = self.model.chat(
                    self.tokenizer,
                    image_path,
                    **chat_kwargs
                )
            
            # Build result
            result = {
                'text': text,
                'type': ocr_type,
                'has_text': bool(text and len(text.strip()) > 0),
                'image_path': image_path,
                'options': cache_kwargs
            }
            
            # Cache the result
            self._save_to_cache(image_path, ocr_type, result, **cache_kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            return {
                'text': '',
                'type': ocr_type,
                'has_text': False,
                'error': str(e),
                'image_path': image_path
            }
    
    def extract_text_batch(
        self,
        image_paths: List[Union[str, Path]],
        ocr_type: str = "ocr",
        show_progress: bool = True
    ) -> Dict[str, Dict]:
        """
        Extract text from multiple images.
        
        Args:
            image_paths: List of image paths
            ocr_type: OCR type ('ocr' or 'format')
            show_progress: Whether to show progress bar
            
        Returns:
            Dict mapping image paths to OCR results
        """
        from tqdm import tqdm
        
        results = {}
        image_paths = [str(p) for p in image_paths]
        
        # Check cache for all images
        uncached_paths = []
        for path in image_paths:
            cached = self._load_from_cache(path, ocr_type)
            if cached is not None:
                results[path] = cached
            else:
                uncached_paths.append(path)
        
        if not uncached_paths:
            logger.info(f"All {len(image_paths)} OCR results loaded from cache")
            return results
        
        logger.info(f"Extracting text from {len(uncached_paths)} uncached images")
        
        # Process uncached images
        iterator = uncached_paths
        if show_progress:
            iterator = tqdm(iterator, desc="OCR extraction")
        
        for path in iterator:
            result = self.extract_text(path, ocr_type=ocr_type)
            results[path] = result
        
        return results
    
    def extract_from_frames(
        self,
        frames_data: List[Dict],
        ocr_type: str = "ocr",
        path_key: str = "path",
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Extract text from frames and add to frame metadata.
        
        Args:
            frames_data: List of frame dicts from VideoSampler
            ocr_type: OCR type
            path_key: Key to access image path in frame dicts
            show_progress: Whether to show progress
            
        Returns:
            Updated frames_data with 'ocr_text' and 'has_text' keys
        """
        # Extract paths
        paths = [frame[path_key] for frame in frames_data if path_key in frame]
        
        if not paths:
            logger.warning("No frame paths found in frames_data")
            return frames_data
        
        # Extract text
        ocr_results = self.extract_text_batch(paths, ocr_type, show_progress)
        
        # Add to frame data
        for frame in frames_data:
            path = frame.get(path_key)
            if path and path in ocr_results:
                ocr_data = ocr_results[path]
                frame['ocr_text'] = ocr_data.get('text', '')
                frame['has_text'] = ocr_data.get('has_text', False)
                frame['ocr_type'] = ocr_type
        
        return frames_data
    
    def has_significant_text(
        self,
        image_path: Union[str, Path],
        min_chars: int = 10
    ) -> bool:
        """
        Quick check if image has significant text content.
        
        Args:
            image_path: Path to image
            min_chars: Minimum character count to consider significant
            
        Returns:
            True if image has significant text
        """
        result = self.extract_text(image_path, ocr_type="ocr")
        text = result.get('text', '')
        return len(text.strip()) >= min_chars


# Backwards compatibility alias
class OCRTool(OCRExtractor):
    """Alias for backwards compatibility."""
    def __init__(self, use_gpu: bool = False, **kwargs):
        device = 'cuda' if use_gpu else None
        super().__init__(device=device, **kwargs)
    
    def batch_ocr(self, image_paths: List[str]) -> Dict[str, str]:
        """Legacy batch OCR method."""
        results = self.extract_text_batch(image_paths, ocr_type="ocr", show_progress=False)
        return {path: res.get('text', '') for path, res in results.items()}


if __name__ == "__main__":
    # Test
    import sys
    import glob
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        frames_dir = sys.argv[1]
        image_paths = glob.glob(f"{frames_dir}/*.jpg") + glob.glob(f"{frames_dir}/*.png")
        
        if not image_paths:
            print(f"No images found in {frames_dir}")
            sys.exit(1)
        
        print("=" * 80)
        print("OCR EXTRACTION TEST")
        print("=" * 80)
        print(f"\nFound {len(image_paths)} images")
        
        # Initialize OCR
        ocr = OCRExtractor(
            model_name="stepfun-ai/GOT-OCR2_0",
            cache_dir="./cache/test_ocr",
            use_cache=True
        )
        
        # Test on first few images
        test_images = image_paths[:3]
        results = ocr.extract_text_batch(test_images, ocr_type="ocr")
        
        print("\nOCR Results:")
        for path, result in results.items():
            filename = Path(path).name
            text = result.get('text', '')
            has_text = result.get('has_text', False)
            
            print(f"\n{filename}:")
            print(f"  Has text: {has_text}")
            if text:
                # Show first 100 chars
                preview = text[:100] + ("..." if len(text) > 100 else "")
                print(f"  Text: {preview}")
        
        # Test cache
        print("\nTesting cache...")
        results2 = ocr.extract_text_batch(test_images[:1], ocr_type="ocr")
        print("Cache working!" if results2 else "Cache failed!")
        
    else:
        print("Usage: python ocr.py <frames_directory>")

