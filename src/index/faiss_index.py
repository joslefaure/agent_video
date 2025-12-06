"""
FAISS-based indexing for efficient frame retrieval.

Provides kNN search over visual embeddings and optional text-based filtering.
"""

import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    FAISS index for efficient similarity search over frame embeddings.
    
    Features:
    - Fast kNN search using FAISS
    - Metadata storage and retrieval
    - Index persistence (save/load)
    - Support for both Flat and HNSW indices
    - Text-based filtering (if OCR metadata available)
    """
    
    def __init__(
        self, 
        dimension: int = 1024,  # DINOv2-large dimension
        index_type: str = "Flat",
        metric: str = "L2"
    ):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimensionality
            index_type: 'Flat' (exact search) or 'HNSW' (approximate, faster)
            metric: 'L2' (Euclidean) or 'IP' (inner product/cosine)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.metadata = []  # Parallel list to store metadata for each vector
        
        # Create index based on type
        if index_type == "Flat":
            if metric == "L2":
                self.index = faiss.IndexFlatL2(dimension)
            else:  # IP
                self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "HNSW":
            # HNSW parameters: M=32 (connections per layer), efConstruction=200
            if metric == "L2":
                self.index = faiss.IndexHNSWFlat(dimension, 32)
            else:
                self.index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 100  # Search-time parameter
        else:
            raise ValueError(f"Unknown index_type: {index_type}")
        
        logger.info(f"Initialized {index_type} FAISS index (dim={dimension}, metric={metric})")
        
        logger.info(f"Initialized {index_type} FAISS index (dim={dimension}, metric={metric})")
    
    def add_items(
        self, 
        embeddings: np.ndarray, 
        meta: List[Dict],
        normalize: bool = False
    ):
        """
        Add embeddings and metadata to the index.
        
        Args:
            embeddings: Array of shape (N, dimension)
            meta: List of metadata dicts for each embedding
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
        """
        if len(embeddings) != len(meta):
            raise ValueError(
                f"Embeddings and metadata length mismatch: {len(embeddings)} vs {len(meta)}"
            )
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.dimension}, got {embeddings.shape[1]}"
            )
        
        # Convert to float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')
        
        # Normalize if using inner product for cosine similarity
        if normalize or self.metric == "IP":
            faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        self.metadata.extend(meta)
        
        logger.info(f"Added {len(embeddings)} items to index. Total: {self.index.ntotal}")
    
    def add_from_frames(
        self,
        frames_data: List[Dict],
        embedding_key: str = "embedding",
        normalize: bool = False
    ):
        """
        Add frames from VideoSampler/VisualEmbedder output.
        
        Args:
            frames_data: List of frame dicts with embeddings
            embedding_key: Key to access embeddings in frame dicts
            normalize: Whether to normalize embeddings
        """
        # Extract embeddings and metadata
        embeddings = []
        metadata = []
        
        for frame in frames_data:
            if embedding_key in frame:
                embeddings.append(frame[embedding_key])
                # Store relevant metadata (exclude large embedding from metadata)
                meta = {k: v for k, v in frame.items() if k != embedding_key and k != 'frame'}
                metadata.append(meta)
        
        if not embeddings:
            logger.warning("No embeddings found in frames_data")
            return
        
        embeddings = np.array(embeddings)
        self.add_items(embeddings, metadata, normalize=normalize)

    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 5,
        normalize: bool = False,
        filter_fn: Optional[callable] = None
    ) -> List[Dict]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_vector: Query embedding (1D array)
            k: Number of results to return
            normalize: Whether to normalize query vector
            filter_fn: Optional function to filter results: fn(metadata) -> bool
            
        Returns:
            List of metadata dicts with added 'score' and 'rank' keys
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Prepare query
        query_vector = query_vector.reshape(1, -1).astype('float32')
        
        if normalize or self.metric == "IP":
            faiss.normalize_L2(query_vector)
        
        # Search
        # If we need filtering, get more results than k
        search_k = k * 10 if filter_fn else k
        search_k = min(search_k, self.index.ntotal)
        
        distances, indices = self.index.search(query_vector, search_k)
        
        # Build results
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1 or idx >= len(self.metadata):
                continue
            
            item = self.metadata[idx].copy()
            
            # Add score (convert distance to similarity if needed)
            if self.metric == "L2":
                item['score'] = float(dist)  # Lower is better
                item['similarity'] = 1.0 / (1.0 + dist)  # Convert to 0-1
            else:  # IP
                item['score'] = float(dist)  # Higher is better
                item['similarity'] = float(dist)
            
            item['rank'] = rank
            item['index'] = int(idx)
            
            # Apply filter if provided
            if filter_fn is None or filter_fn(item):
                results.append(item)
                
                if len(results) >= k:
                    break
        
        return results
    
    def search_by_text(
        self,
        query_text: str,
        k: int = 5,
        text_key: str = "ocr_text"
    ) -> List[Dict]:
        """
        Search by text matching in metadata (requires OCR).
        
        Args:
            query_text: Text to search for
            k: Number of results
            text_key: Metadata key containing text
            
        Returns:
            List of matching metadata dicts
        """
        query_lower = query_text.lower()
        results = []
        
        for idx, meta in enumerate(self.metadata):
            text = meta.get(text_key, "")
            if isinstance(text, str) and query_lower in text.lower():
                item = meta.copy()
                item['index'] = idx
                item['text_match'] = text
                results.append(item)
                
                if len(results) >= k:
                    break
        
        return results
    
    def get_by_indices(self, indices: List[int]) -> List[Dict]:
        """Get metadata for specific indices."""
        results = []
        for idx in indices:
            if 0 <= idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item['index'] = idx
                results.append(item)
        return results
    
    def get_all_metadata(self) -> List[Dict]:
        """Get all metadata."""
        return [m.copy() for m in self.metadata]
    
    def save(self, path: Union[str, Path]):
        """
        Save index and metadata to disk.
        
        Args:
            path: Base path (without extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = str(path) + ".index"
        faiss.write_index(self.index, index_path)
        
        # Save metadata as JSON (if serializable) or pickle
        meta_path = str(path) + ".meta.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'metric': self.metric
            }, f)
        
        logger.info(f"Saved index to {index_path} and metadata to {meta_path}")
    
    def load(self, path: Union[str, Path]):
        """
        Load index and metadata from disk.
        
        Args:
            path: Base path (without extension)
        """
        path = Path(path)
        
        # Load FAISS index
        index_path = str(path) + ".index"
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        meta_path = str(path) + ".meta.pkl"
        if Path(meta_path).exists():
            with open(meta_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.dimension = data['dimension']
                self.index_type = data.get('index_type', 'Flat')
                self.metric = data.get('metric', 'L2')
        else:
            logger.warning(f"Metadata file not found: {meta_path}")
            self.metadata = []
        
        logger.info(f"Loaded index from {index_path} ({self.index.ntotal} items)")
    
    def clear(self):
        """Clear the index and metadata."""
        self.index.reset()
        self.metadata = []
        logger.info("Cleared index")
    
    def __len__(self):
        """Return number of items in index."""
        return self.index.ntotal
    
    def __repr__(self):
        return (f"FAISSIndex(type={self.index_type}, metric={self.metric}, "
                f"dim={self.dimension}, items={len(self)})")


def build_index_from_video(
    video_path: str,
    cache_dir: str = "./cache",
    index_type: str = "Flat",
    normalize: bool = True
) -> Tuple[FAISSIndex, List[Dict]]:
    """
    Build FAISS index from video (end-to-end helper).
    
    Args:
        video_path: Path to video file
        cache_dir: Cache directory for frames and embeddings
        index_type: FAISS index type
        normalize: Whether to normalize embeddings
        
    Returns:
        Tuple of (FAISSIndex, frames_data)
    """
    from .sampler import VideoSampler
    from .embedders import VisualEmbedder
    
    logger.info(f"Building index for video: {video_path}")
    
    # Sample frames
    sampler = VideoSampler(cache_dir=cache_dir, deduplicate=True)
    frames = sampler.sample_keyframes(video_path, num_frames_per_shot=1)
    logger.info(f"Sampled {len(frames)} frames")
    
    # Extract embeddings
    embedder = VisualEmbedder(
        model_name="facebook/dinov2-large",
        cache_dir=f"{cache_dir}/embeddings",
        use_cache=True
    )
    frames = embedder.embed_frames(frames, batch_size=32, show_progress=True)
    
    # Build index
    embedding_dim = embedder.get_embedding_dim()
    index = FAISSIndex(dimension=embedding_dim, index_type=index_type, metric="IP")
    index.add_from_frames(frames, normalize=normalize)
    
    return index, frames


if __name__ == "__main__":
    # Test
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        
        print("=" * 80)
        print("FAISS INDEX TEST")
        print("=" * 80)
        
        # Build index
        index, frames = build_index_from_video(
            video_path,
            cache_dir="./cache/test",
            index_type="Flat",
            normalize=True
        )
        
        print(f"\n{index}")
        print(f"Frames: {len(frames)}")
        
        # Test search with first frame
        if frames:
            query_emb = frames[0]['embedding']
            results = index.search(query_emb, k=5, normalize=True)
            
            print("\nTop 5 similar frames to first frame:")
            for i, res in enumerate(results):
                print(f"  {i+1}. {res.get('frame_id', 'N/A')} "
                      f"(similarity: {res['similarity']:.3f}, "
                      f"timestamp: {res.get('timestamp', 0):.2f}s)")
        
        # Save index
        save_path = "./cache/test/index"
        index.save(save_path)
        print(f"\nSaved index to {save_path}")
        
        # Test load
        index2 = FAISSIndex(dimension=index.dimension)
        index2.load(save_path)
        print(f"Loaded index: {index2}")
        
    else:
        print("Usage: python faiss_index.py <video_path>")

