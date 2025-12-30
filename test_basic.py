#!/usr/bin/env python
"""Basic tests for Fun-ASR-Nano batch inference."""
import os
import sys
import time
import tempfile
from pathlib import Path

# Model path
MODEL_DIR = "/data/extra/models/models--FunAudioLLM--Fun-ASR-Nano-2512/snapshots/a7088d620f755dcdca575b63db184c3ad55b2865"

# Test audio files from workspace
TEST_AUDIOS = [
    "/data/workspace/1/TS-049_自然男声/16000003.wav",
    "/data/workspace/1/TS-047_夹子女声/14000005.wav",
    "/data/workspace/1/TS-041_威严霸总/08000006.wav",
    "/data/workspace/1/TS-044_温柔姐姐/11000008.wav",
    "/data/workspace/1/TS-046_傲娇学姐/13100010.wav",
]


def test_model_load():
    """Test model loading with batch support."""
    from funasr import AutoModel
    
    print("=" * 60)
    print("Test 1: Model Loading")
    print("=" * 60)
    
    start = time.time()
    model = AutoModel(
        model=MODEL_DIR,
        remote_code="./model_batch.py",
        device="cuda:0",
        trust_remote_code=True,
    )
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    assert model is not None, "Model failed to load"
    print("PASS: Model loaded successfully")
    return model


def test_single_inference(model):
    """Test single file inference."""
    print("\n" + "=" * 60)
    print("Test 2: Single File Inference")
    print("=" * 60)
    
    audio_file = TEST_AUDIOS[0]
    print(f"Audio: {audio_file}")
    
    start = time.time()
    result = model.generate(
        input=audio_file,
        language="中文",
        use_itn=True,
        batch_size=1,
        hotwords=[],
    )
    elapsed = time.time() - start
    
    text = result[0].get("text", "") if result else ""
    print(f"Result: {text[:100]}..." if len(text) > 100 else f"Result: {text}")
    print(f"Time: {elapsed:.3f}s")
    
    assert len(text) > 0, "No transcription result"
    print("PASS: Single inference works")
    return elapsed


def test_batch_inference(model, batch_sizes=[2, 3, 5]):
    """Test batch inference with different batch sizes."""
    print("\n" + "=" * 60)
    print("Test 3: Batch Inference")
    print("=" * 60)
    
    results = {}
    
    for bs in batch_sizes:
        audios = TEST_AUDIOS[:bs]
        print(f"\nBatch size {bs}:")
        
        start = time.time()
        batch_results = model.generate(
            input=audios,
            language="中文",
            use_itn=True,
            batch_size=bs,
            hotwords=[],
        )
        batch_time = time.time() - start
        
        # Verify results
        assert len(batch_results) == bs, f"Expected {bs} results, got {len(batch_results)}"
        for i, r in enumerate(batch_results):
            text = r.get("text", "")
            assert len(text) > 0, f"Empty result for file {i}"
            print(f"  [{i}] {text[:50]}...")
        
        results[bs] = batch_time
        print(f"  Time: {batch_time:.3f}s ({batch_time/bs:.3f}s per file)")
    
    print("\nPASS: Batch inference works for all batch sizes")
    return results


def test_serial_vs_batch(model):
    """Compare serial vs batch performance."""
    print("\n" + "=" * 60)
    print("Test 4: Serial vs Batch Performance")
    print("=" * 60)
    
    audios = TEST_AUDIOS[:5]
    
    # Serial processing
    print("Running serial (one by one)...")
    serial_start = time.time()
    for audio in audios:
        model.generate(
            input=audio,
            language="中文",
            use_itn=True,
            batch_size=1,
            hotwords=[],
        )
    serial_time = time.time() - serial_start
    
    # Batch processing
    print("Running batch (all at once)...")
    batch_start = time.time()
    model.generate(
        input=audios,
        language="中文",
        use_itn=True,
        batch_size=len(audios),
        hotwords=[],
    )
    batch_time = time.time() - batch_start
    
    speedup = serial_time / batch_time
    
    print(f"\nResults for {len(audios)} files:")
    print(f"  Serial:  {serial_time:.3f}s ({serial_time/len(audios):.3f}s per file)")
    print(f"  Batch:   {batch_time:.3f}s ({batch_time/len(audios):.3f}s per file)")
    print(f"  Speedup: {speedup:.2f}x")
    
    assert speedup > 1.0, "Batch should be faster than serial"
    print("PASS: Batch is faster than serial")
    return {"serial": serial_time, "batch": batch_time, "speedup": speedup}


def test_empty_input(model):
    """Test handling of empty/invalid inputs."""
    print("\n" + "=" * 60)
    print("Test 5: Error Handling")
    print("=" * 60)
    
    # Test non-existent file
    try:
        model.generate(
            input="/nonexistent/file.wav",
            language="中文",
            batch_size=1,
            hotwords=[],
        )
        print("WARN: Should have raised error for non-existent file")
    except Exception as e:
        print(f"PASS: Correctly raised error for non-existent file: {type(e).__name__}")
    
    # Test empty list
    try:
        result = model.generate(
            input=[],
            language="中文",
            batch_size=1,
            hotwords=[],
        )
        print(f"PASS: Empty list returns: {result}")
    except Exception as e:
        print(f"PASS: Empty list raises error: {type(e).__name__}")


def main():
    """Run all tests."""
    print("Fun-ASR-Nano Basic Tests")
    print(f"Model: {MODEL_DIR}")
    print(f"Test audios: {len(TEST_AUDIOS)} files")
    print()
    
    # Check test files exist
    missing = [f for f in TEST_AUDIOS if not os.path.exists(f)]
    if missing:
        print(f"ERROR: Missing test files: {missing}")
        sys.exit(1)
    
    try:
        model = test_model_load()
        test_single_inference(model)
        test_batch_inference(model)
        perf = test_serial_vs_batch(model)
        test_empty_input(model)
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        print(f"\nPerformance Summary:")
        print(f"  Batch speedup: {perf['speedup']:.2f}x")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
