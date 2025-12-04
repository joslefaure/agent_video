import pytest
import os
import json
import numpy as np
from src.tools.sampler import VideoSampler
from src.tools.ocr import OCRTool
from src.tools.embedders import VisualEmbedder
from src.index.faiss_index import FAISSIndex
from src.orchestrator.vllm_agent import VLLMAgent

# Constants
SAMPLE_VIDEO = "sample_data/tiny_sample.mp4"

@pytest.fixture(scope="session", autouse=True)
def setup_sample_data():
    # Create dummy video if not exists
    if not os.path.exists(SAMPLE_VIDEO):
        import cv2
        os.makedirs(os.path.dirname(SAMPLE_VIDEO), exist_ok=True)
        height, width = 240, 320
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(SAMPLE_VIDEO, fourcc, 30, (width, height))
        for i in range(60):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(img, "Test", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(img)
        out.release()

def test_sampler():
    sampler = VideoSampler(cache_dir="./tests/cache")
    info = sampler.get_video_info(SAMPLE_VIDEO)
    assert info['duration'] > 0
    
    frames = sampler.sample_keyframes(SAMPLE_VIDEO)
    assert len(frames) > 0
    assert os.path.exists(frames[0]['path'])

def test_ocr():
    ocr = OCRTool()
    # We don't have a real text image, but it should run without error
    # and return empty string or dummy text depending on implementation
    res = ocr.extract_text("non_existent.jpg")
    assert isinstance(res, str)

def test_embedder():
    embedder = VisualEmbedder()
    # Create a dummy image
    dummy_img_path = "tests/dummy.jpg"
    from PIL import Image
    Image.new('RGB', (100, 100)).save(dummy_img_path)
    
    emb = embedder.embed_image(dummy_img_path)
    assert emb.shape == (384,) # Assuming DINOv2 small default

def test_faiss():
    index = FAISSIndex(dimension=384)
    emb = np.random.rand(1, 384).astype('float32')
    meta = [{"id": 1}]
    index.add_items(emb, meta)
    
    results = index.search(emb[0])
    assert len(results) > 0
    assert results[0]['id'] == 1

def test_agent_smoke():
    agent = VLLMAgent(mock=True)
    res = agent.run(SAMPLE_VIDEO, "What is in the video?", ["A", "B"])
    assert "answer_text" in res
    assert "confidence" in res
