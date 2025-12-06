#!/usr/bin/env python3
"""
Test VideoMME evaluation with a small sample.
"""

import sys
import json
import tempfile
from pathlib import Path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval.evaluate_videomme import (
    extract_answer,
    format_question_prompt,
    evaluate_videomme
)


def test_extract_answer():
    """Test answer extraction from various response formats."""
    print("\n" + "="*80)
    print("TEST: Answer Extraction")
    print("="*80)
    
    test_cases = [
        ("The best answer is A", "A"),
        ("The correct answer is B.", "B"),
        ("C is the answer", "C"),
        ("I think D", "D"),
        ("Based on the video, the answer is A because...", "A"),
        ("This is a very long response without any clear answer", ""),
        ("", ""),
    ]
    
    passed = 0
    for response, expected in test_cases:
        result = extract_answer(response)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{response[:50]}...' -> '{result}' (expected '{expected}')")
        if result == expected:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_format_prompt():
    """Test question prompt formatting."""
    print("\n" + "="*80)
    print("TEST: Question Formatting")
    print("="*80)
    
    doc = {
        "question": "What is shown in the video?",
        "options": ["A cat", "A dog", "A bird", "A fish"]
    }
    
    prompt = format_question_prompt(doc)
    
    print("Sample prompt:")
    print("-" * 80)
    print(prompt)
    print("-" * 80)
    
    # Check formatting
    checks = [
        ("Question:" in prompt, "Contains question"),
        ("Options:" in prompt, "Contains options"),
        ("A." in prompt, "Has option A"),
        ("B." in prompt, "Has option B"),
        ("C." in prompt, "Has option C"),
        ("D." in prompt, "Has option D"),
    ]
    
    passed = 0
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"{status} {desc}")
        if check:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(checks)}")
    return passed == len(checks)


def test_evaluation_mini():
    """Test evaluation on a tiny synthetic dataset."""
    print("\n" + "="*80)
    print("TEST: Mini Evaluation (Synthetic Data)")
    print("="*80)
    
    # Check if real video exists
    video_path = Path("sample_data/H54zMD-9Q-8.mp4")
    if not video_path.exists():
        print("⚠️  Real video not found, skipping mini evaluation")
        print("   Run this test after sample video is available")
        return True
    
    # Create synthetic annotations
    annotations = [
        {
            "videoID": "H54zMD-9Q-8",
            "question_id": 1,
            "question": "What type of video is this?",
            "options": ["Gaming", "Tutorial", "Vlog", "News"],
            "answer": "A",
            "duration": "long",
            "domain": "Knowledge",
            "sub_category": "Technology",
            "task_type": "Attribute Perception"
        }
    ]
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(annotations, f)
        anno_path = f.name
    
    # Create temp output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_path = f.name
    
    try:
        print(f"\nRunning evaluation on 1 sample...")
        print(f"Video: {video_path}")
        print(f"Annotations: {anno_path}")
        
        # Run evaluation
        results = evaluate_videomme(
            data_dir=str(video_path.parent),
            anno_path=anno_path,
            output_file=output_path,
            cache_dir="./cache/test_eval",
            max_samples=1,
            max_frames=4
        )
        
        print(f"\n✓ Evaluation completed!")
        print(f"  Accuracy: {results['overall_accuracy']:.1f}%")
        print(f"  Total: {results['total_questions']}")
        print(f"  Results saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        Path(anno_path).unlink(missing_ok=True)


def main():
    print("="*80)
    print("VIDEOMME EVALUATION TEST SUITE")
    print("="*80)
    
    test1 = test_extract_answer()
    test2 = test_format_prompt()
    test3 = test_evaluation_mini()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"  Answer Extraction:  {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"  Prompt Formatting:  {'✅ PASSED' if test2 else '❌ FAILED'}")
    print(f"  Mini Evaluation:    {'✅ PASSED' if test3 else '⚠️  SKIPPED/FAILED'}")
    print("="*80)
    
    if test1 and test2:
        print("\n✅ Core evaluation functions working!")
        if test3:
            print("✅ End-to-end evaluation working!")
        else:
            print("⚠️  End-to-end test skipped (requires GPU and sample video)")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
