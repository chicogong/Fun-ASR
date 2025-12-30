#!/usr/bin/env python
"""Throughput test for Fun-ASR-Nano HTTP API."""
import os
import sys
import time
import requests
import concurrent.futures
from pathlib import Path

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Test audio files
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
]


def get_audio_duration(file_path):
    """Get audio duration in seconds (approximate from file size for WAV)."""
    try:
        import wave
        with wave.open(file_path, 'rb') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except:
        # Fallback: estimate ~16kHz 16-bit mono
        size = os.path.getsize(file_path)
        return size / (16000 * 2)


def transcribe_single(audio_path):
    """Send single file to /transcribe endpoint."""
    start = time.time()
    with open(audio_path, 'rb') as f:
        response = requests.post(
            f"{API_URL}/transcribe",
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            data={"language": "中文"},
        )
    elapsed = time.time() - start
    
    if response.status_code != 200:
        return {"error": response.text, "time": elapsed}
    
    return {"text": response.json().get("text", ""), "time": elapsed}


def transcribe_batch(audio_paths):
    """Send multiple files to /transcribe_batch endpoint."""
    start = time.time()
    files = []
    for path in audio_paths:
        files.append(("files", (os.path.basename(path), open(path, 'rb'), "audio/wav")))
    
    response = requests.post(
        f"{API_URL}/transcribe_batch",
        files=files,
        data={"language": "中文"},
    )
    elapsed = time.time() - start
    
    # Close file handles
    for _, (_, f, _) in files:
        f.close()
    
    if response.status_code != 200:
        return {"error": response.text, "time": elapsed}
    
    return {"results": response.json().get("results", []), "time": elapsed}


def test_serial_throughput(num_files=10):
    """Test throughput with serial requests."""
    print(f"\n{'='*60}")
    print(f"Serial Throughput Test ({num_files} files)")
    print(f"{'='*60}")
    
    audios = (TEST_AUDIOS * ((num_files // len(TEST_AUDIOS)) + 1))[:num_files]
    total_duration = sum(get_audio_duration(a) for a in audios)
    
    start = time.time()
    results = []
    for i, audio in enumerate(audios):
        r = transcribe_single(audio)
        results.append(r)
        if "error" in r:
            print(f"  [{i+1}] ERROR: {r['error'][:50]}")
        else:
            print(f"  [{i+1}] {r['time']:.2f}s - {r['text'][:30]}...")
    
    total_time = time.time() - start
    
    print(f"\nResults:")
    print(f"  Total audio duration: {total_duration:.1f}s")
    print(f"  Processing time: {total_time:.2f}s")
    print(f"  RTF (Real-Time Factor): {total_time / total_duration:.3f}")
    print(f"  Throughput: {num_files / total_time:.2f} files/sec")
    print(f"  Throughput: {total_duration / total_time:.2f}x realtime")
    
    return {"files": num_files, "time": total_time, "audio_duration": total_duration}


def test_batch_throughput(batch_size=5, num_batches=2):
    """Test throughput with batch requests."""
    print(f"\n{'='*60}")
    print(f"Batch Throughput Test (batch_size={batch_size}, num_batches={num_batches})")
    print(f"{'='*60}")
    
    num_files = batch_size * num_batches
    audios = (TEST_AUDIOS * ((num_files // len(TEST_AUDIOS)) + 1))[:num_files]
    total_duration = sum(get_audio_duration(a) for a in audios)
    
    start = time.time()
    for i in range(num_batches):
        batch = audios[i*batch_size : (i+1)*batch_size]
        r = transcribe_batch(batch)
        if "error" in r:
            print(f"  Batch {i+1}: ERROR - {r['error'][:50]}")
        else:
            print(f"  Batch {i+1}: {r['time']:.2f}s for {len(r['results'])} files")
    
    total_time = time.time() - start
    
    print(f"\nResults:")
    print(f"  Total audio duration: {total_duration:.1f}s")
    print(f"  Processing time: {total_time:.2f}s")
    print(f"  RTF (Real-Time Factor): {total_time / total_duration:.3f}")
    print(f"  Throughput: {num_files / total_time:.2f} files/sec")
    print(f"  Throughput: {total_duration / total_time:.2f}x realtime")
    
    return {"files": num_files, "time": total_time, "audio_duration": total_duration}


def test_concurrent_throughput(num_workers=3, files_per_worker=3):
    """Test throughput with concurrent requests (simulates multiple users)."""
    print(f"\n{'='*60}")
    print(f"Concurrent Throughput Test ({num_workers} workers x {files_per_worker} files)")
    print(f"{'='*60}")
    
    num_files = num_workers * files_per_worker
    audios = (TEST_AUDIOS * ((num_files // len(TEST_AUDIOS)) + 1))[:num_files]
    total_duration = sum(get_audio_duration(a) for a in audios)
    
    def worker(worker_id, audio_list):
        results = []
        for audio in audio_list:
            r = transcribe_single(audio)
            results.append(r)
        return worker_id, results
    
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            worker_audios = audios[i*files_per_worker : (i+1)*files_per_worker]
            futures.append(executor.submit(worker, i, worker_audios))
        
        for future in concurrent.futures.as_completed(futures):
            worker_id, results = future.result()
            errors = sum(1 for r in results if "error" in r)
            times = [r["time"] for r in results if "time" in r]
            print(f"  Worker {worker_id+1}: {len(results)} files, avg {sum(times)/len(times):.2f}s, {errors} errors")
    
    total_time = time.time() - start
    
    print(f"\nResults:")
    print(f"  Total audio duration: {total_duration:.1f}s")
    print(f"  Processing time: {total_time:.2f}s")
    print(f"  RTF (Real-Time Factor): {total_time / total_duration:.3f}")
    print(f"  Throughput: {num_files / total_time:.2f} files/sec")
    print(f"  Throughput: {total_duration / total_time:.2f}x realtime")
    
    return {"files": num_files, "time": total_time, "audio_duration": total_duration}


def main():
    """Run all throughput tests."""
    print("Fun-ASR-Nano Throughput Tests")
    print(f"API: {API_URL}")
    
    # Check API is up
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code != 200 or not r.json().get("model_loaded"):
            print("ERROR: API not ready")
            sys.exit(1)
        print("API is ready.\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to API - {e}")
        sys.exit(1)
    
    # Run tests
    serial_result = test_serial_throughput(num_files=5)
    batch_result = test_batch_throughput(batch_size=5, num_batches=1)
    concurrent_result = test_concurrent_throughput(num_workers=3, files_per_worker=2)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'Files':<8} {'Time':<10} {'Files/s':<10} {'RTF':<10}")
    print(f"{'-'*60}")
    
    for name, r in [("Serial", serial_result), ("Batch", batch_result), ("Concurrent", concurrent_result)]:
        files_per_sec = r["files"] / r["time"]
        rtf = r["time"] / r["audio_duration"]
        print(f"{name:<20} {r['files']:<8} {r['time']:<10.2f} {files_per_sec:<10.2f} {rtf:<10.3f}")
    
    print(f"\nNote: RTF < 1.0 means faster than realtime")


if __name__ == "__main__":
    main()
