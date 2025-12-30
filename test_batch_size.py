#!/usr/bin/env python
"""Test different batch sizes to find optimal throughput."""
import os
import time
import torch

MODEL_DIR = "/data/extra/models/models--FunAudioLLM--Fun-ASR-Nano-2512/snapshots/a7088d620f755dcdca575b63db184c3ad55b2865"

# 20 test audio files
TEST_AUDIOS = [
    "/data/workspace/1/TS-049_自然男声/16000003.wav",
    "/data/workspace/1/TS-047_夹子女声/14000005.wav",
    "/data/workspace/1/TS-041_威严霸总/08000006.wav",
    "/data/workspace/1/TS-044_温柔姐姐/11000008.wav",
    "/data/workspace/1/TS-046_傲娇学姐/13100010.wav",
    "/data/workspace/1/TS-049_自然男声/16000001.wav",
    "/data/workspace/1/TS-047_夹子女声/14000006.wav",
    "/data/workspace/1/TS-041_威严霸总/08000007.wav",
    "/data/workspace/1/TS-044_温柔姐姐/11000009.wav",
    "/data/workspace/1/TS-046_傲娇学姐/13100008.wav",
    "/data/workspace/1/TS-049_自然男声/16000002.wav",
    "/data/workspace/1/TS-047_夹子女声/14000007.wav",
    "/data/workspace/1/TS-041_威严霸总/08000008.wav",
    "/data/workspace/1/TS-044_温柔姐姐/11000010.wav",
    "/data/workspace/1/TS-046_傲娇学姐/13100009.wav",
    "/data/workspace/1/TS-049_自然男声/16000004.wav",
    "/data/workspace/1/TS-047_夹子女声/14000008.wav",
    "/data/workspace/1/TS-041_威严霸总/08000009.wav",
    "/data/workspace/1/TS-044_温柔姐姐/11000011.wav",
    "/data/workspace/1/TS-046_傲娇学姐/13100007.wav",
]


def get_gpu_memory():
    """Get GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def test_batch_size(model, batch_size, audios):
    """Test a specific batch size."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start = time.time()
    try:
        results = model.generate(
            input=audios[:batch_size],
            language="中文",
            use_itn=True,
            batch_size=batch_size,
            hotwords=[],
        )
        elapsed = time.time() - start
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return {
            "success": True,
            "time": elapsed,
            "per_file": elapsed / batch_size,
            "peak_mem_mb": peak_mem,
            "files": batch_size,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)[:100],
        }


def main():
    print("=" * 70)
    print("Batch Size Optimization Test")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Test files: {len(TEST_AUDIOS)}")
    print()
    
    # Load model
    print("Loading model...")
    from funasr import AutoModel
    model = AutoModel(
        model=MODEL_DIR,
        remote_code="./model_batch.py",
        device="cuda:0",
        trust_remote_code=True,
        disable_update=True,
    )
    
    base_mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Model loaded. Base memory: {base_mem:.0f} MB")
    print()
    
    # First, test serial (batch_size=1) as baseline
    print("Testing batch_size=1 (baseline)...")
    serial_times = []
    for i, audio in enumerate(TEST_AUDIOS[:5]):
        start = time.time()
        model.generate(input=audio, language="中文", batch_size=1, hotwords=[])
        serial_times.append(time.time() - start)
    baseline_per_file = sum(serial_times) / len(serial_times)
    print(f"Baseline (serial): {baseline_per_file:.3f}s per file")
    print()
    
    # Test different batch sizes
    batch_sizes = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    results = []
    
    print(f"{'Batch':<8} {'Time':<10} {'Per File':<12} {'Speedup':<10} {'Peak Mem':<12} {'Status'}")
    print("-" * 70)
    
    for bs in batch_sizes:
        if bs > len(TEST_AUDIOS):
            break
            
        result = test_batch_size(model, bs, TEST_AUDIOS)
        
        if result["success"]:
            speedup = baseline_per_file / result["per_file"]
            print(f"{bs:<8} {result['time']:<10.2f} {result['per_file']:<12.3f} {speedup:<10.2f}x {result['peak_mem_mb']:<12.0f} OK")
            results.append({
                "batch_size": bs,
                "speedup": speedup,
                "per_file": result["per_file"],
                "peak_mem": result["peak_mem_mb"],
            })
        else:
            print(f"{bs:<8} {'--':<10} {'--':<12} {'--':<10} {'--':<12} FAIL: {result['error'][:30]}")
            break  # Stop if we hit memory limit
    
    # Find optimal
    if results:
        best = max(results, key=lambda x: x["speedup"])
        print()
        print("=" * 70)
        print(f"OPTIMAL BATCH SIZE: {best['batch_size']}")
        print(f"  Speedup: {best['speedup']:.2f}x vs serial")
        print(f"  Time per file: {best['per_file']:.3f}s")
        print(f"  Peak GPU memory: {best['peak_mem']:.0f} MB")
        print("=" * 70)
        
        # Calculate throughput improvement
        print()
        print("Throughput comparison:")
        print(f"  Serial (batch=1):  {1/baseline_per_file:.2f} files/sec")
        print(f"  Optimal (batch={best['batch_size']}): {1/best['per_file']:.2f} files/sec")
        print(f"  Improvement: {best['speedup']:.2f}x")


if __name__ == "__main__":
    main()
