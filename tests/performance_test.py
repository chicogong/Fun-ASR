#!/usr/bin/env python3
"""
Fun-ASR MLT Performance Testing Script
Tests RTF, throughput, and daily processing capacity
"""

import requests
import time
import statistics
import os
import sys
from pathlib import Path
import json
import subprocess
import tempfile

# Configuration
API_BASE_URL = os.environ.get("API_URL", "http://localhost:8000")
TEST_AUDIO_DIR = os.environ.get("TEST_AUDIO_DIR", "./test_audio")
LANGUAGE = "zh"
MAX_BATCH_SIZE = 20

class PerformanceTest:
    def __init__(self, api_url=API_BASE_URL):
        self.api_url = api_url
        self.results = {
            "single_file": [],
            "batch": {},
            "system_info": {}
        }

    def check_health(self):
        """Check if service is healthy"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Service healthy: {response.json()}")
                return True
            else:
                print(f"❌ Service unhealthy: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to service: {e}")
            return False

    def get_audio_duration(self, audio_path):
        """Get audio duration using ffprobe"""
        try:
            cmd = [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of",
                "default=noprint_wrappers=1:nokey=1", audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"⚠️  Cannot get duration for {audio_path}: {e}")
            # Estimate based on file size (rough estimate: 16kHz, 16bit, mono ≈ 32KB/s)
            file_size = os.path.getsize(audio_path)
            return file_size / 32000  # Rough estimate

    def generate_test_audio(self, duration_seconds=10, count=1):
        """Generate test audio files using ffmpeg"""
        test_files = []
        temp_dir = tempfile.mkdtemp(prefix="funasr_test_")

        for i in range(count):
            output_path = os.path.join(temp_dir, f"test_audio_{i+1}_{duration_seconds}s.wav")

            # Generate sine wave audio (440Hz tone)
            cmd = [
                "ffmpeg", "-f", "lavfi", "-i",
                f"sine=frequency=440:duration={duration_seconds}",
                "-ar", "16000", "-ac", "1", "-y", output_path
            ]

            try:
                subprocess.run(cmd, capture_output=True, timeout=30, check=True)
                test_files.append(output_path)
                print(f"✅ Generated test audio: {output_path}")
            except Exception as e:
                print(f"⚠️  Failed to generate audio {i+1}: {e}")

        return test_files, temp_dir

    def find_test_audio_files(self):
        """Find existing audio files for testing"""
        test_dir = Path(TEST_AUDIO_DIR)
        if not test_dir.exists():
            print(f"⚠️  Test directory {TEST_AUDIO_DIR} not found")
            return []

        audio_files = []
        for ext in ["*.wav", "*.mp3", "*.m4a", "*.flac"]:
            audio_files.extend(test_dir.glob(ext))

        return [str(f) for f in audio_files]

    def test_single_file(self, audio_path):
        """Test single file transcription"""
        try:
            duration = self.get_audio_duration(audio_path)

            start_time = time.time()
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
                data = {"language": LANGUAGE}
                response = requests.post(
                    f"{self.api_url}/transcribe",
                    files=files,
                    data=data,
                    timeout=120
                )
            end_time = time.time()

            processing_time = end_time - start_time
            rtf = processing_time / duration if duration > 0 else 0

            result = {
                "file": os.path.basename(audio_path),
                "duration": duration,
                "processing_time": processing_time,
                "rtf": rtf,
                "success": response.status_code == 200,
                "response": response.json() if response.status_code == 200 else response.text
            }

            self.results["single_file"].append(result)
            return result

        except Exception as e:
            print(f"❌ Error testing {audio_path}: {e}")
            return None

    def test_batch(self, audio_paths, batch_size):
        """Test batch transcription"""
        if batch_size not in self.results["batch"]:
            self.results["batch"][batch_size] = []

        try:
            # Calculate total duration
            total_duration = sum(self.get_audio_duration(p) for p in audio_paths)

            start_time = time.time()
            files = [
                ("files", (os.path.basename(p), open(p, "rb"), "audio/wav"))
                for p in audio_paths
            ]
            data = {"language": LANGUAGE}

            response = requests.post(
                f"{self.api_url}/transcribe_batch",
                files=files,
                data=data,
                timeout=300
            )

            # Close file handles
            for _, file_tuple in files:
                file_tuple[1].close()

            end_time = time.time()

            processing_time = end_time - start_time
            rtf = processing_time / total_duration if total_duration > 0 else 0
            throughput = len(audio_paths) / processing_time if processing_time > 0 else 0

            result = {
                "batch_size": batch_size,
                "num_files": len(audio_paths),
                "total_duration": total_duration,
                "processing_time": processing_time,
                "rtf": rtf,
                "throughput_files_per_sec": throughput,
                "throughput_hours_per_day": (total_duration / processing_time * 86400) / 3600 if processing_time > 0 else 0,
                "success": response.status_code == 200,
                "response": response.json() if response.status_code == 200 else response.text
            }

            self.results["batch"][batch_size].append(result)
            return result

        except Exception as e:
            print(f"❌ Error testing batch size {batch_size}: {e}")
            return None

    def get_system_info(self):
        """Get system information"""
        info = {}

        # GPU info
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(", ")
                info["gpu"] = {
                    "name": gpu_info[0],
                    "memory_total_mb": gpu_info[1],
                    "memory_used_mb": gpu_info[2],
                    "utilization": gpu_info[3]
                }
        except:
            info["gpu"] = "N/A"

        # CPU info
        try:
            result = subprocess.run(["nproc"], capture_output=True, text=True, timeout=5)
            info["cpu_cores"] = int(result.stdout.strip())
        except:
            info["cpu_cores"] = "N/A"

        # Service info
        try:
            response = requests.get(f"{self.api_url}/info", timeout=5)
            if response.status_code == 200:
                info["service"] = response.json()
        except:
            info["service"] = "N/A"

        self.results["system_info"] = info
        return info

    def print_results(self):
        """Print formatted results"""
        print("\n" + "="*80)
        print("📊 PERFORMANCE TEST RESULTS")
        print("="*80)

        # System info
        print("\n🖥️  System Information:")
        print("-" * 80)
        if "gpu" in self.results["system_info"] and self.results["system_info"]["gpu"] != "N/A":
            gpu = self.results["system_info"]["gpu"]
            print(f"GPU: {gpu['name']}")
            print(f"GPU Memory: {gpu['memory_used_mb']} / {gpu['memory_total_mb']} MB")
            print(f"GPU Utilization: {gpu['utilization']}%")
        print(f"CPU Cores: {self.results['system_info'].get('cpu_cores', 'N/A')}")

        if "service" in self.results["system_info"]:
            print(f"Service: {self.results['system_info']['service']}")

        # Single file results
        if self.results["single_file"]:
            print("\n📄 Single File Transcription:")
            print("-" * 80)
            rtfs = [r["rtf"] for r in self.results["single_file"] if r["success"]]
            durations = [r["duration"] for r in self.results["single_file"] if r["success"]]
            proc_times = [r["processing_time"] for r in self.results["single_file"] if r["success"]]

            if rtfs:
                print(f"Number of tests: {len(rtfs)}")
                print(f"Average audio duration: {statistics.mean(durations):.2f}s")
                print(f"Average processing time: {statistics.mean(proc_times):.2f}s")
                print(f"Average RTF: {statistics.mean(rtfs):.4f}")
                print(f"Min RTF: {min(rtfs):.4f}")
                print(f"Max RTF: {max(rtfs):.4f}")
                print(f"Median RTF: {statistics.median(rtfs):.4f}")

                # Estimate daily capacity
                avg_rtf = statistics.mean(rtfs)
                if avg_rtf > 0:
                    hours_per_day = 24 / avg_rtf
                    print(f"\n💡 Estimated daily capacity (single file): {hours_per_day:.2f} hours of audio")

        # Batch results
        if self.results["batch"]:
            print("\n📦 Batch Transcription:")
            print("-" * 80)

            for batch_size in sorted(self.results["batch"].keys()):
                results = [r for r in self.results["batch"][batch_size] if r["success"]]
                if not results:
                    continue

                print(f"\nBatch Size: {batch_size}")
                print(f"  Number of tests: {len(results)}")

                rtfs = [r["rtf"] for r in results]
                throughputs = [r["throughput_hours_per_day"] for r in results]

                print(f"  Average RTF: {statistics.mean(rtfs):.4f}")
                print(f"  Min RTF: {min(rtfs):.4f}")
                print(f"  Max RTF: {max(rtfs):.4f}")
                print(f"  Average throughput: {statistics.mean([r['throughput_files_per_sec'] for r in results]):.2f} files/sec")
                print(f"  💡 Estimated daily capacity: {statistics.mean(throughputs):.2f} hours of audio")

        print("\n" + "="*80)

    def save_results(self, output_file="performance_results.json"):
        """Save results to JSON file"""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {output_file}")

def main():
    print("🚀 Fun-ASR MLT Performance Testing")
    print("=" * 80)

    tester = PerformanceTest()

    # Check health
    if not tester.check_health():
        print("❌ Service not available. Please start the service first.")
        sys.exit(1)

    # Get system info
    print("\n📊 Collecting system information...")
    tester.get_system_info()

    # Find or generate test audio
    print("\n🔍 Looking for test audio files...")
    test_files = tester.find_test_audio_files()

    temp_dir = None
    if not test_files:
        print("⚠️  No existing audio files found. Generating test files...")
        # Generate test files with different durations
        durations = [5, 10, 30, 60]  # seconds
        all_generated = []
        for duration in durations:
            generated, temp_dir = tester.generate_test_audio(duration, count=5)
            all_generated.extend(generated)
        test_files = all_generated
        print(f"✅ Generated {len(test_files)} test files")
    else:
        print(f"✅ Found {len(test_files)} existing audio files")
        # Limit to first 20 files for testing
        test_files = test_files[:20]

    if not test_files:
        print("❌ No test files available")
        sys.exit(1)

    # Test single file
    print(f"\n🔬 Testing single file transcription ({len(test_files[:5])} files)...")
    for audio_file in test_files[:5]:  # Test first 5 files
        print(f"\nTesting: {os.path.basename(audio_file)}")
        result = tester.test_single_file(audio_file)
        if result:
            print(f"  Duration: {result['duration']:.2f}s")
            print(f"  Processing: {result['processing_time']:.2f}s")
            print(f"  RTF: {result['rtf']:.4f}")
            if result['success']:
                print(f"  ✅ Success")
            else:
                print(f"  ❌ Failed")

    # Test batch with different sizes
    print(f"\n🔬 Testing batch transcription...")
    batch_sizes = [1, 2, 5, 10, 20]

    for batch_size in batch_sizes:
        if batch_size > len(test_files):
            print(f"⚠️  Skipping batch size {batch_size} (not enough test files)")
            continue

        print(f"\n📦 Testing batch size: {batch_size}")
        # Run test 3 times for each batch size
        for i in range(3):
            batch_files = test_files[:batch_size]
            print(f"  Run {i+1}/3...")
            result = tester.test_batch(batch_files, batch_size)
            if result:
                print(f"    Files: {result['num_files']}")
                print(f"    Total duration: {result['total_duration']:.2f}s")
                print(f"    Processing time: {result['processing_time']:.2f}s")
                print(f"    RTF: {result['rtf']:.4f}")
                print(f"    Throughput: {result['throughput_files_per_sec']:.2f} files/sec")
                print(f"    Daily capacity: {result['throughput_hours_per_day']:.2f} hours")
            time.sleep(1)  # Small delay between tests

    # Print summary
    tester.print_results()

    # Save results
    tester.save_results()

    # Cleanup temp files
    if temp_dir and os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\n🧹 Cleaned up temporary files")

    print("\n✅ Performance testing complete!")

if __name__ == "__main__":
    main()
