#!/usr/bin/env python3
"""Quick verification script for metrics implementation."""

import sys
sys.path.insert(0, '.')

from src.training.metrics import WERMetric

def test_basic():
    """Test basic WER computation."""
    metric = WERMetric()
    
    # Test perfect match
    wer = metric.compute(["hello world"], ["hello world"])
    assert wer == 0.0, f"Expected WER=0.0, got {wer}"
    print("✓ Perfect match test passed")
    
    # Test with errors
    wer = metric.compute(["the cat"], ["the dog"])
    assert 0.0 < wer < 1.0, f"Expected 0 < WER < 1, got {wer}"
    print(f"✓ Error test passed (WER={wer:.4f})")
    
    # Test return_details
    result = metric.compute(["the cat"], ["the dog"], return_details=True)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert 'wer' in result
    assert 'insertions' in result
    assert 'deletions' in result
    assert 'substitutions' in result
    assert 'hits' in result
    print(f"✓ Detailed metrics test passed: {result}")
    
    print("\n✅ All verification tests passed!")

if __name__ == "__main__":
    test_basic()
