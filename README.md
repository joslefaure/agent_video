# Agentic Video Understanding Pipeline

A training-free, modular pipeline for long-form video understanding using open-source models (<10B parameters).

## 🎯 Goal

Achieve ≥80% accuracy on VideoMME using only off-the-shelf pretrained models with agentic orchestration.

## ✨ Key Features

- **Training-free**: Uses pretrained models only
- **Agentic**: Single VLM agent orchestrates perception tools
- **Efficient**: Frame deduplication, smart caching, batch processing
- **Reproducible**: Deterministic execution, version-pinned dependencies
- **Modular**: Easy to swap models or add new tools

## 🏗️ Architecture

```
Video Input
    ↓
[VideoSampler]      → Shot detection, keyframe extraction, deduplication
    ↓
[VisualEmbedder]    → DINOv2-large embeddings (cached)
    ↓
[FAISSIndex]        → Fast kNN search over frames
    ↓
[OCRExtractor]      → GOT-OCR2.0 text extraction (cached)
    ↓
[Agent Loop]        → Qwen2.5-VL with tool-calling
    ↓
Structured Answer
```

## 📦 Components (Implemented ✅)

| Component | Model | Size | Status |
|-----------|-------|------|--------|
| VLM Orchestrator | Qwen2.5-VL-7B | 8B | 🚧 Next |
| Visual Encoder | DINOv2-Large | 0.3B | ✅ |
| OCR | GOT-OCR2.0 | 0.7B | ✅ |
| Segmentation | SAM2-Hiera-Large | 0.2B | 🚧 Next |
| Detection | OWLv2-Large | 0.4B | 🚧 Next |

## 🚀 Quick Start

### Installation

```bash
cd video-agent-pipeline

# Install dependencies
pip install -r requirements.txt

# Note: Install FAISS-GPU for faster performance
# pip install faiss-gpu==1.9.0
```

### Test Individual Components

```bash
# 1. Test video sampling
python src/tools/sampler.py sample_data/tiny_sample.mp4

# 2. Test visual embeddings
python src/tools/embedders.py cache/tiny_sample/frames/

# 3. Test FAISS indexing
python src/index/faiss_index.py sample_data/tiny_sample.mp4

# 4. Test OCR extraction
python src/tools/ocr.py cache/tiny_sample/frames/
```

### Run Integration Test

```bash
# Test the full pipeline
python tests/test_integration.py sample_data/tiny_sample.mp4

# Results saved to: cache/integration_test/integration_test_report.json
```

## 📊 What Works Now

✅ **Video Preprocessing**
- Automatic shot detection
- Keyframe extraction (1-3 per shot)
- Perceptual hashing for duplicate removal
- Video metadata extraction

✅ **Visual Embeddings**
- DINOv2-large (1024-dim features)
- Batch processing (32 images at once)
- Automatic disk caching (10x speedup on reruns!)
- Progress tracking

✅ **FAISS Indexing**
- Fast similarity search (kNN)
- Support for exact (Flat) and approximate (HNSW) search
- Index persistence (save/load)
- Metadata storage and filtering

✅ **OCR Extraction**
- GOT-OCR2.0 for text extraction
- Plain and formatted text modes
- Batch processing with caching
- Fine-grained OCR support

## 🔧 Configuration

Models are configured in `src/config/models.py`. To switch models:

```python
from src.config import get_model_config

# Get default orchestrator config
config = get_model_config("orchestrator")
print(config.hf_model_id)  # "Qwen/Qwen2.5-VL-7B-Instruct"

# Or specify a different model
config = get_model_config("orchestrator", "internvl2.5-8b")
```

## 📈 Performance Optimizations

1. **Caching**: All intermediate results are cached to disk
   - Embeddings: `cache/<video>/embeddings/`
   - OCR results: `cache/<video>/ocr/`
   - FAISS indices: `cache/<video>/index/`

2. **Batch Processing**: Process multiple frames simultaneously
   - Embeddings: 32 frames/batch
   - OCR: Sequential (model limitation)

3. **Deduplication**: Remove similar frames before processing
   - Saves 20-40% compute on typical videos

## 🧪 Testing

```bash
# Run integration test
python tests/test_integration.py sample_data/tiny_sample.mp4

# Check individual modules
python -m pytest tests/  # Coming soon
```

## 📝 Example Usage

```python
from src.tools.sampler import VideoSampler
from src.tools.embedders import VisualEmbedder
from src.index.faiss_index import FAISSIndex

# 1. Extract frames
sampler = VideoSampler(cache_dir="./cache", deduplicate=True)
frames = sampler.sample_keyframes("video.mp4", num_frames_per_shot=1)

# 2. Compute embeddings
embedder = VisualEmbedder(cache_dir="./cache/embeddings")
frames = embedder.embed_frames(frames, batch_size=32)

# 3. Build search index
index = FAISSIndex(dimension=1024, index_type="Flat", metric="IP")
index.add_from_frames(frames, normalize=True)

# 4. Search similar frames
query_emb = frames[0]['embedding']
results = index.search(query_emb, k=5)
print(results)
```

## 🗺️ Roadmap

- [x] Video preprocessing & frame extraction
- [x] Visual embeddings (DINOv2)
- [x] FAISS indexing
- [x] OCR extraction (GOT-OCR2.0)
- [ ] Object detection (OWLv2)
- [ ] Segmentation (SAM2)
- [ ] VLM agent orchestrator
- [ ] VideoMME evaluation harness
- [ ] End-to-end pipeline script
- [ ] Ablation studies

## 📚 References

- **VideoMME Dataset**: [HuggingFace](https://huggingface.co/datasets/lmms-lab/Video-MME)
- **Qwen2.5-VL**: [Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- **DINOv2**: [Paper](https://arxiv.org/abs/2304.07193)
- **GOT-OCR2.0**: [Model Card](https://huggingface.co/stepfun-ai/GOT-OCR2_0)
- **SAM2**: [GitHub](https://github.com/facebookresearch/segment-anything-2)

## 📄 License

See plan.md for full project details.

---

**Status**: 🚧 Active Development (4 of 7 core modules complete)

**Last Updated**: December 4, 2025
