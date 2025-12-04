import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple

class FAISSIndex:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = [] # Parallel list to store metadata for each vector

    def add_items(self, embeddings: np.ndarray, meta: List[Dict]):
        if len(embeddings) != len(meta):
            raise ValueError("Embeddings and metadata must have same length")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {embeddings.shape[1]}")

        self.index.add(embeddings.astype('float32'))
        self.metadata.extend(meta)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
            
        query_vector = query_vector.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item['score'] = float(dist)
                results.append(item)
        
        return results

    def save(self, path: str):
        faiss.write_index(self.index, path + ".index")
        # Save metadata separately (e.g. json)
        pass

    def load(self, path: str):
        self.index = faiss.read_index(path + ".index")
        # Load metadata
        pass
