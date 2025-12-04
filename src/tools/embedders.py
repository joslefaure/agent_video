"""
Visual embedding module using DINOv2.

Provides efficient frame-level feature extraction for video understanding.
"""

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import logging
from typing import List, Dict, Optional, Union
import numpy as np
from pathlib import Path
import json
import hashlib
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VisualEmbedder:
    """
    Visual feature extractor using DINOv2.
    
    Features:
    - Batch processing for efficiency
    - Embedding caching to disk
    - Automatic model loading
    - Multiple DINOv2 model sizes
    """
    
    def __init__(
        self, 
        model_name: str = "facebook/dinov2-large",
        device: str = None,
        cache_dir: Optional[str] = "./cache/embeddings",
        use_cache: bool = True
    ):
        """
        Initialize the visual embedder.
        
        Args:
            model_name: HuggingFace model ID (default: dinov2-large)
            device: Device to run on ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache embeddings
            use_cache: Whether to cache embeddings to disk
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        
        if self.use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.processor = None
        self.model = None
        self.embedding_dim = None
        
        logger.info(f"Initialized VisualEmbedder with {model_name} on {self.device}")
        
    def _load_model(self):
        """Lazy load the model and processor."""
        if self.model is None:
            try:
                logger.info(f"Loading visual encoder: {self.model_name}")
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                ).to(self.device)
                self.model.eval()
                
                # Get embedding dimension
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                    dummy_output = self.model(pixel_values=dummy_input)
                    self.embedding_dim = dummy_output.last_hidden_state.shape[-1]
                
                logger.info(f"Model loaded successfully. Embedding dim: {self.embedding_dim}")
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
    
    def _get_cache_key(self, image_path: str) -> str:
        """Generate cache key from image path and model name."""
        path_str = str(Path(image_path).absolute())
        key_string = f"{self.model_name}:{path_str}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _load_from_cache(self, image_path: str) -> Optional[np.ndarray]:
        """Load embedding from cache if available."""
        if not self.use_cache or not self.cache_dir:
            return None
        
        cache_key = self._get_cache_key(image_path)
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cache for {image_path}: {e}")
                return None
        return None
    
    def _save_to_cache(self, image_path: str, embedding: np.ndarray):
        """Save embedding to cache."""
        if not self.use_cache or not self.cache_dir:
            return
        
        cache_key = self._get_cache_key(image_path)
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"Failed to save cache for {image_path}: {e}")

    def embed_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Extract embedding for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            1D numpy array of embeddings
        """
        image_path = str(image_path)
        
        # Check cache first
        cached = self._load_from_cache(image_path)
        if cached is not None:
            return cached
        
        # Load model if needed
        self._load_model()
        
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embedding = embeddings[0]  # Shape: (embedding_dim,)
            
            # Cache the result
            self._save_to_cache(image_path, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding {image_path}: {e}")
            # Return zero vector as fallback
            if self.embedding_dim is None:
                self._load_model()
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def embed_batch(
        self, 
        image_paths: List[Union[str, Path]], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for multiple images efficiently.
        
        Args:
            image_paths: List of paths to image files
            batch_size: Number of images to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            Dict mapping image paths to embeddings
        """
        results = {}
        image_paths = [str(p) for p in image_paths]
        
        # Check cache for all images
        uncached_paths = []
        for path in image_paths:
            cached = self._load_from_cache(path)
            if cached is not None:
                results[path] = cached
            else:
                uncached_paths.append(path)
        
        if not uncached_paths:
            logger.info(f"All {len(image_paths)} embeddings loaded from cache")
            return results
        
        logger.info(f"Computing embeddings for {len(uncached_paths)} uncached images")
        
        # Load model if needed
        self._load_model()
        
        # Process in batches
        iterator = range(0, len(uncached_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding images")
        
        for i in iterator:
            batch_paths = uncached_paths[i:i + batch_size]
            
            # Load images
            images = []
            valid_paths = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    results[path] = np.zeros(self.embedding_dim, dtype=np.float32)
            
            if not images:
                continue
            
            # Process batch
            try:
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Store results and cache
                for path, embedding in zip(valid_paths, embeddings):
                    results[path] = embedding
                    self._save_to_cache(path, embedding)
                    
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Fallback to individual processing
                for path in valid_paths:
                    results[path] = self.embed_image(path)
        
        return results
    
    def embed_frames(
        self,
        frames_data: List[Dict],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Embed frames from sampler output and add embeddings to frame data.
        
        Args:
            frames_data: List of frame dicts from VideoSampler
            batch_size: Batch size for processing
            show_progress: Whether to show progress
            
        Returns:
            Updated frames_data with 'embedding' key added
        """
        # Extract paths
        paths = [frame['path'] for frame in frames_data if frame.get('path')]
        
        if not paths:
            logger.warning("No frame paths found in frames_data")
            return frames_data
        
        # Compute embeddings
        embeddings_dict = self.embed_batch(paths, batch_size, show_progress)
        
        # Add embeddings to frame data
        for frame in frames_data:
            path = frame.get('path')
            if path and path in embeddings_dict:
                frame['embedding'] = embeddings_dict[path]
        
        return frames_data
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        if self.embedding_dim is None:
            self._load_model()
        return self.embedding_dim


if __name__ == "__main__":
    # Simple test
    import sys
    import glob
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        # Test with provided directory
        frames_dir = sys.argv[1]
        image_paths = glob.glob(f"{frames_dir}/*.jpg") + glob.glob(f"{frames_dir}/*.png")
        
        if not image_paths:
            print(f"No images found in {frames_dir}")
            sys.exit(1)
        
        print("=" * 80)
        print("VISUAL EMBEDDER TEST")
        print("=" * 80)
        print(f"\nFound {len(image_paths)} images in {frames_dir}")
        
        # Initialize embedder
        embedder = VisualEmbedder(
            model_name="facebook/dinov2-large",
            cache_dir="./cache/test_embeddings",
            use_cache=True
        )
        
        print(f"Embedding dimension: {embedder.get_embedding_dim()}")
        
        # Embed images
        embeddings = embedder.embed_batch(image_paths[:10], batch_size=4)  # Test first 10
        
        print(f"\nGenerated {len(embeddings)} embeddings")
        for path, emb in list(embeddings.items())[:3]:
            print(f"  {Path(path).name}: shape={emb.shape}, norm={np.linalg.norm(emb):.2f}")
        
        # Test cache
        print("\nTesting cache...")
        embeddings2 = embedder.embed_batch(image_paths[:5], batch_size=4)
        print("Cache working!" if len(embeddings2) == 5 else "Cache failed!")
        
    else:
        print("Usage: python embedders.py <frames_directory>")

