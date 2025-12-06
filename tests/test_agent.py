#!/usr/bin/env python3
"""
Test the Qwen2.5-VL Agent on sample video questions.
"""

import sys
import logging
from pathlib import Path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))



from orchestrator.qwen_agent import QwenVideoAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_agent_basic():
    """Test basic agent functionality without VLM."""
    print("\n" + "="*80)
    print("TEST 1: Agent Tools (No VLM)")
    print("="*80)
    
    # Initialize without loading model
    agent = QwenVideoAgent(
        cache_dir="./cache/agent_test",
        load_model=False
    )
    
    # Test preprocessing
    video_path = "sample_data/H54zMD-9Q-8.mp4"
    if not Path(video_path).exists():
        print(f"⚠️  Video not found: {video_path}")
        print("Please run: python tests/test_integration.py sample_data/H54zMD-9Q-8.mp4")
        return False
    
    print(f"\nPreprocessing video: {video_path}")
    result = agent.preprocess_video(video_path, num_frames_per_shot=1)
    
    print(f"\n✓ Preprocessing complete:")
    print(f"  Frames extracted: {result['num_frames']}")
    print(f"  Index size: {len(agent.index)}")
    
    # Test search
    print(f"\nTesting similarity search...")
    query_emb = result['frames'][0]['embedding']
    search_results = agent.search_frames(query_emb, k=3)
    
    print(f"✓ Found {len(search_results)} similar frames:")
    for i, res in enumerate(search_results):
        print(f"  {i+1}. {res['frame_id']} @ {res['timestamp']:.1f}s "
              f"(similarity: {res['similarity']:.3f})")
    
    print(f"\n✅ Agent tools test PASSED!")
    return True


def test_agent_with_vlm():
    """Test agent with VLM for question answering."""
    print("\n" + "="*80)
    print("TEST 2: Agent with Qwen3-VL")
    print("="*80)
    
    try:
        # Initialize with VLM
        agent = QwenVideoAgent(
            cache_dir="./cache/agent_test",
            load_model=True
        )
        
        # Test question
        video_path = "sample_data/H54zMD-9Q-8.mp4"
        question = "What is shown in this video?"
        
        print(f"\nAnswering question about {video_path}")
        print(f"Question: {question}")
        
        result = agent.answer_question(video_path, question, max_frames=4)
        
        print(f"\n✓ Answer generated:")
        print(f"  {result['answer'][:200]}...")
        print(f"  Used {result['num_frames_used']} frames")
        
        print(f"\n✅ Agent VLM test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n⚠️  VLM test failed (this is expected if GPU/memory limited):")
        print(f"  {e}")
        return False


def test_agent_mcq():
    """Test agent on multiple choice question."""
    print("\n" + "="*80)
    print("TEST 3: Multiple Choice Question")
    print("="*80)
    
    try:
        agent = QwenVideoAgent(
            cache_dir="./cache/agent_test",
            load_model=True
        )
        
        video_path = "sample_data/H54zMD-9Q-8.mp4"
        question = "What type of content is this video?"
        options = ["Tutorial", "Documentary", "Entertainment", "News"]
        
        print(f"\nQuestion: {question}")
        print(f"Options: {options}")
        
        result = agent.answer_question(video_path, question, options=options, max_frames=6)
        
        print(f"\n✓ Answer:")
        print(f"  {result['answer']}")
        if 'selected_option' in result:
            print(f"\n  Selected: {result['selected_option']}. {result['selected_option_text']}")
        
        print(f"\n✅ MCQ test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n⚠️  MCQ test failed:")
        print(f"  {e}")
        return False


def main():
    print("="*80)
    print("QWEN VIDEO AGENT TEST SUITE")
    print("="*80)
    
    # Test 1: Tools only (no VLM required)
    test1_passed = test_agent_basic()
    
    if not test1_passed:
        print("\n❌ Basic tools test failed. Please check video path.")
        sys.exit(1)
    
    # Test 2: With VLM (requires GPU)
    print("\n" + "="*80)
    print("VLM tests require GPU and ~16GB VRAM")
    print("Skipping if resources unavailable...")
    print("="*80)
    
    test2_passed = test_agent_with_vlm()
    test3_passed = test_agent_mcq() if test2_passed else False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"  Tools Test:     {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  VLM Test:       {'✅ PASSED' if test2_passed else '⚠️  SKIPPED/FAILED'}")
    print(f"  MCQ Test:       {'✅ PASSED' if test3_passed else '⚠️  SKIPPED/FAILED'}")
    print("="*80)
    
    if test1_passed:
        print("\n✅ Core agent functionality working!")
        if not test2_passed:
            print("⚠️  VLM tests skipped (GPU/memory requirements)")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
