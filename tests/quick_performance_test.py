#!/usr/bin/env python3
"""
Fun-ASR MLT Quick Performance Test
Tests RTF and throughput without external dependencies
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

def test_single_file(audio_path, duration):
    """Test single file transcription"""
    start_time = time.time()

    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
        data = {"language": LANGUAGE}
        response = requests.post(f"{API_URL}/transcribe", files=files, data=data, timeout=120)

    processing_time = time.time() - start_time
    rtf = processing_time / duration if duration > 0 else 0

    return {
        "duration": duration,
        "processing_time": processing_time,
        "rtf": rtf,
        "success": response.status_code == 200,
        "text": response.json().get("text", "") if response.status_code == 200 else ""
    }

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
    print("🚀 Fun-ASR MLT Performance Test")
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
        print(f"📊 Service Info: {response.json()}")
    except:
        pass

    # Create temp directory for test files
    temp_dir = tempfile.mkdtemp(prefix="funasr_perf_")
    print(f"\n📁 Using temp directory: {temp_dir}")

    # Generate test files with different durations
    test_durations = [5, 10, 15, 20, 30]  # seconds
    test_files = []
    file_durations = []

    print("\n🎵 Generating test audio files...")
    for i, duration in enumerate(test_durations):
        filepath = os.path.join(temp_dir, f"test_{i+1}_{duration}s.wav")
        generate_wav(filepath, duration)
        test_files.append(filepath)
        file_durations.append(duration)
        print(f"  ✅ Generated {duration}s audio: {os.path.basename(filepath)}")

    # ========== SINGLE FILE TESTS ==========
    print("\n" + "="*80)
    print("📄 SINGLE FILE TRANSCRIPTION TEST")
    print("="*80)

    single_results = []
    for audio_file, duration in zip(test_files, file_durations):
        print(f"\n🔬 Testing: {os.path.basename(audio_file)} ({duration}s)")
        result = test_single_file(audio_file, duration)
        single_results.append(result)

        print(f"  Duration: {result['duration']:.2f}s")
        print(f"  Processing: {result['processing_time']:.2f}s")
        print(f"  RTF: {result['rtf']:.4f}")
        print(f"  Status: {'✅ Success' if result['success'] else '❌ Failed'}")

    # Single file statistics
    rtfs = [r['rtf'] for r in single_results if r['success']]
    if rtfs:
        print("\n📊 Single File Statistics:")
        print(f"  Tests: {len(rtfs)}")
        print(f"  Average RTF: {statistics.mean(rtfs):.4f}")
        print(f"  Min RTF: {min(rtfs):.4f}")
        print(f"  Max RTF: {max(rtfs):.4f}")
        print(f"  Median RTF: {statistics.median(rtfs):.4f}")

        avg_rtf = statistics.mean(rtfs)
        if avg_rtf > 0:
            hours_per_day = 24 / avg_rtf
            print(f"\n💡 Estimated Daily Capacity (Single File):")
            print(f"  {hours_per_day:.2f} hours of audio per day")
            print(f"  {hours_per_day * 60:.0f} minutes per day")

    # ========== BATCH TESTS ==========
    print("\n" + "="*80)
    print("📦 BATCH TRANSCRIPTION TEST")
    print("="*80)

    batch_results = {}
    batch_sizes = [1, 2, 3, 5]

    for batch_size in batch_sizes:
        if batch_size > len(test_files):
            continue

        print(f"\n🔬 Testing Batch Size: {batch_size}")
        batch_tests = []

        # Run 3 tests for each batch size
        for run in range(3):
            print(f"  Run {run+1}/3...")
            batch_files = test_files[:batch_size]
            batch_durations = file_durations[:batch_size]

            result = test_batch(batch_files, batch_durations)
            batch_tests.append(result)

            print(f"    Files: {result['batch_size']}")
            print(f"    Total duration: {result['total_duration']:.2f}s")
            print(f"    Processing time: {result['processing_time']:.2f}s")
            print(f"    RTF: {result['rtf']:.4f}")
            print(f"    Throughput: {result['throughput_files_per_sec']:.2f} files/sec")
            print(f"    Daily capacity: {result['audio_hours_per_day']:.2f} hours")

            time.sleep(1)  # Small delay between tests

        batch_results[batch_size] = batch_tests

    # Batch statistics
    print("\n" + "="*80)
    print("📊 BATCH STATISTICS")
    print("="*80)

    for batch_size, results in sorted(batch_results.items()):
        success_results = [r for r in results if r['success']]
        if not success_results:
            continue

        rtfs = [r['rtf'] for r in success_results]
        throughputs = [r['audio_hours_per_day'] for r in success_results]

        print(f"\n📦 Batch Size {batch_size}:")
        print(f"  Tests: {len(success_results)}")
        print(f"  Average RTF: {statistics.mean(rtfs):.4f}")
        print(f"  Min RTF: {min(rtfs):.4f}")
        print(f"  Max RTF: {max(rtfs):.4f}")
        print(f"  Average daily capacity: {statistics.mean(throughputs):.2f} hours")
        print(f"  Max daily capacity: {max(throughputs):.2f} hours")

    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("🎯 PERFORMANCE SUMMARY")
    print("="*80)

    if rtfs:
        best_single_rtf = min(rtfs)
        print(f"\n✨ Single File Mode:")
        print(f"  Best RTF: {best_single_rtf:.4f}")
        print(f"  Daily capacity: {24 / best_single_rtf:.2f} hours")

    if batch_results:
        # Find best batch performance
        all_batch_rtfs = []
        all_batch_capacities = []
        for results in batch_results.values():
            for r in results:
                if r['success']:
                    all_batch_rtfs.append(r['rtf'])
                    all_batch_capacities.append(r['audio_hours_per_day'])

        if all_batch_rtfs:
            best_batch_rtf = min(all_batch_rtfs)
            best_capacity = max(all_batch_capacities)
            print(f"\n✨ Batch Mode:")
            print(f"  Best RTF: {best_batch_rtf:.4f}")
            print(f"  Max daily capacity: {best_capacity:.2f} hours")
            print(f"  Max daily capacity: {best_capacity * 60:.0f} minutes")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"\n🧹 Cleaned up temp files")

    print("\n✅ Performance testing complete!")
    print("="*80)

if __name__ == "__main__":
    main()
