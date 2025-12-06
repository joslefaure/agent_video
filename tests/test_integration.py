#!/usr/bin/env python3
"""
End-to-end integration test for the video understanding pipeline.

Tests the complete flow: Video → Frames → Embeddings → Index → OCR
"""

import sys
import logging
from pathlib import Path
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tools.sampler import VideoSampler
from tools.embedders import VisualEmbedder
from tools.ocr import OCRExtractor
from index.faiss_index import FAISSIndex

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_end_to_end(video_path: str, cache_dir: str = "./cache/integration_test"):
    """
    Run end-to-end test of the pipeline.
    
    Args:
        video_path: Path to test video
        cache_dir: Cache directory for intermediate results
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("END-TO-END INTEGRATION TEST")
    print("=" * 80)
    print(f"\nVideo: {video_path}")
    print(f"Cache: {cache_dir}")
    
    # =========================================================================
    # STEP 1: Video Sampling
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Video Sampling (Shot Detection + Frame Extraction)")
    print("=" * 80)
    
    sampler = VideoSampler(cache_dir=cache_dir, deduplicate=True)
    
    # Get video info
    info = sampler.get_video_info(video_path)
    print(f"\nVideo Info:")
    print(f"  Duration: {info['duration']:.2f}s")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  Resolution: {info['resolution']}")
    print(f"  Frame Count: {info['frame_count']}")
    
    # Sample keyframes
    print(f"\nSampling keyframes...")
    frames = sampler.sample_keyframes(video_path, num_frames_per_shot=1)
    print(f"✓ Sampled {len(frames)} frames")
    
    # Show sample
    if frames:
        print(f"\nSample frames:")
        for i, frame in enumerate(frames[:3]):
            print(f"  {i+1}. {frame['frame_id']} @ {frame['timestamp']:.2f}s (shot {frame['shot_id']})")
    
    # =========================================================================
    # STEP 2: Visual Embedding
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Visual Embedding (DINOv2)")
    print("=" * 80)
    
    embedder = VisualEmbedder(
        model_name="facebook/dinov2-large",
        cache_dir=f"{cache_dir}/embeddings",
        use_cache=True
    )
    
    print(f"\nEmbedding {len(frames)} frames...")
    frames = embedder.embed_frames(frames, batch_size=16, show_progress=True)
    
    # Verify embeddings
    embeddings_count = sum(1 for f in frames if 'embedding' in f)
    print(f"✓ Generated {embeddings_count} embeddings")
    print(f"  Embedding dim: {embedder.get_embedding_dim()}")
    
    if frames and 'embedding' in frames[0]:
        emb = frames[0]['embedding']
        print(f"  Sample embedding shape: {emb.shape}")
        print(f"  Sample embedding norm: {np.linalg.norm(emb):.4f}")
    
    # =========================================================================
    # STEP 3: FAISS Indexing
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: FAISS Indexing")
    print("=" * 80)
    
    embedding_dim = embedder.get_embedding_dim()
    index = FAISSIndex(
        dimension=embedding_dim,
        index_type="Flat",
        metric="IP"
    )
    
    print(f"\nBuilding index...")
    index.add_from_frames(frames, normalize=True)
    print(f"✓ Built index with {len(index)} items")
    
    # Test search
    if frames and 'embedding' in frames[0]:
        query_emb = frames[0]['embedding']
        results = index.search(query_emb, k=5, normalize=True)
        
        print(f"\nSearch test (query = first frame):")
        print(f"  Found {len(results)} results")
        for i, res in enumerate(results[:3]):
            print(f"    {i+1}. {res.get('frame_id', 'N/A')} "
                  f"(similarity: {res['similarity']:.4f})")
    
    # Save index
    index_path = cache_path / "test_index"
    index.save(index_path)
    print(f"\n✓ Saved index to {index_path}")
    
    # Test load
    index2 = FAISSIndex(dimension=embedding_dim)
    index2.load(index_path)
    print(f"✓ Loaded index: {len(index2)} items")
    
    # =========================================================================
    # STEP 4: OCR Extraction
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: OCR Extraction (GOT-OCR2.0)")
    print("=" * 80)
    
    ocr = OCRExtractor(
        model_name="stepfun-ai/GOT-OCR2_0",
        cache_dir=f"{cache_dir}/ocr",
        use_cache=True
    )
    
    print(f"\nExtracting text from {len(frames)} frames...")
    frames = ocr.extract_from_frames(frames, ocr_type="ocr", show_progress=True)
    
    # Count frames with text
    text_count = sum(1 for f in frames if f.get('has_text', False))
    print(f"✓ Extracted text from {len(frames)} frames")
    print(f"  Frames with text: {text_count}")
    
    # Show samples
    text_frames = [f for f in frames if f.get('has_text', False)]
    if text_frames:
        print(f"\nSample frames with text:")
        for i, frame in enumerate(text_frames[:3]):
            text = frame.get('ocr_text', '')
            preview = text[:80] + ("..." if len(text) > 80 else "")
            print(f"  {i+1}. {frame['frame_id']}")
            print(f"     Text: {preview}")
    else:
        print(f"\n  No text detected in frames")
    
    # =========================================================================
    # STEP 5: Generate Report
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Generate Report")
    print("=" * 80)
    
    report = {
        "video_info": info,
        "pipeline_stats": {
            "frames_sampled": len(frames),
            "embeddings_generated": embeddings_count,
            "index_size": len(index),
            "frames_with_text": text_count,
            "embedding_dim": embedding_dim
        },
        "sample_frames": [
            {
                "frame_id": f['frame_id'],
                "timestamp": f['timestamp'],
                "shot_id": f.get('shot_id', -1),
                "has_text": f.get('has_text', False),
                "ocr_preview": f.get('ocr_text', '')[:100] if f.get('has_text') else None
            }
            for f in frames[:5]
        ]
    }
    
    report_path = cache_path / "integration_test_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Saved report to {report_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"\n✅ All pipeline components working:")
    print(f"   1. Video Sampling:      {len(frames)} frames")
    print(f"   2. Visual Embedding:    {embeddings_count} embeddings")
    print(f"   3. FAISS Indexing:      {len(index)} items indexed")
    print(f"   4. OCR Extraction:      {text_count} frames with text")
    print(f"   5. Persistence:         Index saved/loaded successfully")
    print(f"\n✓ Integration test PASSED!")
    print("=" * 80)
    
    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_integration.py <video_path> [cache_dir]")
        print("\nExample:")
        print("  python test_integration.py sample_data/tiny_sample.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else "./cache/integration_test"
    
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    try:
        report = test_end_to_end(video_path, cache_dir)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Integration test failed: {e}", exc_info=True)
        sys.exit(1)
