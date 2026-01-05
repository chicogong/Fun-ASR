#!/usr/bin/env python3
"""
Fun-ASR MLT Full Performance Test
Tests batch sizes from 1 to 20 to find maximum throughput
"""

import requests
import time
import statistics
import os
import tempfile
import struct
import wave

API_URL = "http://localhost:8000"
LANGUAGE = "zh"
MAX_BATCH_SIZE = 20

def generate_wav(filepath, duration_sec=10, sample_rate=16000):
    """Generate a simple WAV file (silence)"""
    num_samples = duration_sec * sample_rate

    with wave.open(filepath, 'w') as wav:
        wav.setnchannels(1)  # mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)

        # Generate silence (zeros)
        for _ in range(num_samples):
            wav.writeframes(struct.pack('<h', 0))

    return duration_sec

def test_batch(audio_paths, durations):
    """Test batch transcription"""
    total_duration = sum(durations)

    start_time = time.time()

    files = [("files", (os.path.basename(p), open(p, "rb"), "audio/wav")) for p in audio_paths]
    data = {"language": LANGUAGE}

    response = requests.post(f"{API_URL}/transcribe_batch", files=files, data=data, timeout=300)

    for _, file_tuple in files:
        file_tuple[1].close()

    processing_time = time.time() - start_time
    rtf = processing_time / total_duration if total_duration > 0 else 0
    throughput_files_per_sec = len(audio_paths) / processing_time if processing_time > 0 else 0
    audio_hours_per_day = (total_duration / processing_time * 86400) / 3600 if processing_time > 0 else 0

    return {
        "batch_size": len(audio_paths),
        "total_duration": total_duration,
        "processing_time": processing_time,
        "rtf": rtf,
        "throughput_files_per_sec": throughput_files_per_sec,
        "audio_hours_per_day": audio_hours_per_day,
        "success": response.status_code == 200
    }

def main():
    print("="*80)
    print("🚀 Fun-ASR MLT Full Batch Performance Test (1-20 files)")
    print("="*80)

    # Check health
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"\n✅ Service Status: {response.json()}")
    except:
        print(f"\n❌ Service not available at {API_URL}")
        return

    # Get service info
    try:
        response = requests.get(f"{API_URL}/info", timeout=5)
        info = response.json()
        print(f"📊 Service Info:")
        print(f"  Model: {info.get('model_path', 'N/A')}")
        print(f"  Device: {info.get('device', 'N/A')}")
        print(f"  GPU: {info.get('gpu_name', 'N/A')}")
        print(f"  GPU Memory: {info.get('gpu_memory_used_gb', 0):.1f} / {info.get('gpu_memory_total_gb', 0):.1f} GB")
    except:
        pass

    # Create temp directory for test files
    temp_dir = tempfile.mkdtemp(prefix="funasr_perf_")
    print(f"\n📁 Using temp directory: {temp_dir}")

    # Generate 20 test files (10 seconds each)
    test_files = []
    file_durations = []
    duration = 10  # seconds per file

    print(f"\n🎵 Generating {MAX_BATCH_SIZE} test audio files ({duration}s each)...")
    for i in range(MAX_BATCH_SIZE):
        filepath = os.path.join(temp_dir, f"test_{i+1:02d}.wav")
        generate_wav(filepath, duration)
        test_files.append(filepath)
        file_durations.append(duration)

    print(f"  ✅ Generated {len(test_files)} files, total duration: {sum(file_durations)}s")

    # Test different batch sizes
    print("\n" + "="*80)
    print("📦 BATCH TRANSCRIPTION TEST")
    print("="*80)

    batch_results = {}
    batch_sizes = [1, 2, 3, 4, 5, 8, 10, 12, 15, 18, 20]

    for batch_size in batch_sizes:
        print(f"\n🔬 Testing Batch Size: {batch_size}")
        batch_tests = []

        # Run 3 tests for each batch size
        for run in range(3):
            print(f"  Run {run+1}/3...", end=" ", flush=True)
            batch_files = test_files[:batch_size]
            batch_durations = file_durations[:batch_size]

            result = test_batch(batch_files, batch_durations)
            batch_tests.append(result)

            print(f"RTF: {result['rtf']:.4f}, Capacity: {result['audio_hours_per_day']:.2f} hours/day")

            time.sleep(0.5)  # Small delay between tests

        batch_results[batch_size] = batch_tests

    # Calculate statistics
    print("\n" + "="*80)
    print("📊 DETAILED BATCH STATISTICS")
    print("="*80)

    stats_data = []

    for batch_size, results in sorted(batch_results.items()):
        success_results = [r for r in results if r['success']]
        if not success_results:
            continue

        rtfs = [r['rtf'] for r in success_results]
        capacities = [r['audio_hours_per_day'] for r in success_results]
        throughputs = [r['throughput_files_per_sec'] for r in success_results]

        avg_rtf = statistics.mean(rtfs)
        min_rtf = min(rtfs)
        avg_capacity = statistics.mean(capacities)
        max_capacity = max(capacities)
        avg_throughput = statistics.mean(throughputs)

        stats_data.append({
            "batch_size": batch_size,
            "avg_rtf": avg_rtf,
            "min_rtf": min_rtf,
            "avg_capacity": avg_capacity,
            "max_capacity": max_capacity,
            "avg_throughput": avg_throughput
        })

        print(f"\n📦 Batch Size {batch_size}:")
        print(f"  Average RTF: {avg_rtf:.4f}")
        print(f"  Best RTF: {min_rtf:.4f}")
        print(f"  Average capacity: {avg_capacity:.2f} hours/day")
        print(f"  Max capacity: {max_capacity:.2f} hours/day")
        print(f"  Throughput: {avg_throughput:.2f} files/sec")

    # Find optimal batch size
    print("\n" + "="*80)
    print("🎯 PERFORMANCE SUMMARY")
    print("="*80)

    if stats_data:
        # Find best RTF
        best_rtf_entry = min(stats_data, key=lambda x: x['min_rtf'])
        print(f"\n🏆 Best RTF Performance:")
        print(f"  Batch Size: {best_rtf_entry['batch_size']}")
        print(f"  RTF: {best_rtf_entry['min_rtf']:.4f}")
        print(f"  Daily Capacity: {best_rtf_entry['max_capacity']:.2f} hours")

        # Find best capacity
        best_capacity_entry = max(stats_data, key=lambda x: x['max_capacity'])
        print(f"\n🚀 Maximum Daily Capacity:")
        print(f"  Batch Size: {best_capacity_entry['batch_size']}")
        print(f"  RTF: {best_capacity_entry['min_rtf']:.4f}")
        print(f"  Daily Capacity: {best_capacity_entry['max_capacity']:.2f} hours")
        print(f"  Daily Capacity: {best_capacity_entry['max_capacity'] * 60:.0f} minutes")
        print(f"  Daily Capacity: {best_capacity_entry['max_capacity'] * 3600:.0f} seconds")

        # Find best throughput
        best_throughput_entry = max(stats_data, key=lambda x: x['avg_throughput'])
        print(f"\n⚡ Maximum Throughput:")
        print(f"  Batch Size: {best_throughput_entry['batch_size']}")
        print(f"  Throughput: {best_throughput_entry['avg_throughput']:.2f} files/sec")

        # Show comparison table
        print("\n" + "="*80)
        print("📈 BATCH SIZE COMPARISON TABLE")
        print("="*80)
        print(f"{'Batch':>6} | {'Avg RTF':>8} | {'Best RTF':>9} | {'Avg Hours/Day':>14} | {'Max Hours/Day':>14}")
        print("-" * 80)
        for entry in stats_data:
            print(f"{entry['batch_size']:>6} | {entry['avg_rtf']:>8.4f} | {entry['min_rtf']:>9.4f} | "
                  f"{entry['avg_capacity']:>14.2f} | {entry['max_capacity']:>14.2f}")

        # Recommendations
        print("\n" + "="*80)
        print("💡 RECOMMENDATIONS")
        print("="*80)

        # Find sweet spot (balance between RTF and capacity)
        sweet_spot = min([e for e in stats_data if e['batch_size'] >= 2], key=lambda x: x['avg_rtf'])

        print(f"\n✨ Recommended Batch Size: {sweet_spot['batch_size']}")
        print(f"  Reason: Best balance between RTF and capacity")
        print(f"  RTF: {sweet_spot['avg_rtf']:.4f}")
        print(f"  Daily Capacity: {sweet_spot['avg_capacity']:.2f} hours")

        print(f"\n🎯 For Maximum Throughput: Use batch size {best_capacity_entry['batch_size']}")
        print(f"  Can process up to {best_capacity_entry['max_capacity']:.2f} hours of audio per day")

        print(f"\n⚡ Processing Speed:")
        print(f"  At optimal settings, this system can process audio {1/best_rtf_entry['min_rtf']:.1f}x faster than real-time")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"\n🧹 Cleaned up temp files")

    print("\n✅ Full performance testing complete!")
    print("="*80)

if __name__ == "__main__":
    main()
