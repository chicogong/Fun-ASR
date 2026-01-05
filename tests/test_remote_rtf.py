#!/usr/bin/env python3
"""测试远程服务器的 RTF 性能"""

import requests
import time
import wave
import struct
import tempfile
import os

REMOTE_URL = os.environ.get("REMOTE_URL", "http://localhost:8000")
LOCAL_URL = "http://localhost:8000"

def generate_wav(filepath, duration_sec=10, sample_rate=16000):
    """生成测试音频"""
    num_samples = duration_sec * sample_rate
    with wave.open(filepath, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for _ in range(num_samples):
            wav.writeframes(struct.pack('<h', 0))
    return duration_sec

def test_rtf(url, audio_path, duration):
    """测试单次 RTF"""
    start_time = time.time()
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
        data = {"language": "zh"}
        response = requests.post(f"{url}/transcribe", files=files, data=data, timeout=120)
    processing_time = time.time() - start_time
    rtf = processing_time / duration if duration > 0 else 0
    return rtf, processing_time, response.status_code == 200

def main():
    print("="*80)
    print("🔬 远程 CPU vs 本地 GPU RTF 对比测试")
    print("="*80)

    # 生成测试文件
    temp_dir = tempfile.mkdtemp()
    test_durations = [5, 10, 20, 30]

    print("\n📊 测试配置:")
    print(f"  远程服务器: {REMOTE_URL} (CPU)")
    print(f"  本地服务器: {LOCAL_URL} (GPU)")
    print(f"  测试音频: {test_durations} 秒\n")

    results = {"remote": [], "local": []}

    for duration in test_durations:
        filepath = os.path.join(temp_dir, f"test_{duration}s.wav")
        generate_wav(filepath, duration)

        print(f"\n🎵 测试 {duration} 秒音频:")

        # 测试远程 CPU
        print(f"  远程 CPU: ", end="", flush=True)
        try:
            rtf, proc_time, success = test_rtf(REMOTE_URL, filepath, duration)
            results["remote"].append(rtf)
            print(f"RTF={rtf:.4f}, 处理时间={proc_time:.2f}s {'✅' if success else '❌'}")
        except Exception as e:
            print(f"❌ 失败: {e}")

        # 测试本地 GPU
        print(f"  本地 GPU: ", end="", flush=True)
        try:
            rtf, proc_time, success = test_rtf(LOCAL_URL, filepath, duration)
            results["local"].append(rtf)
            print(f"RTF={rtf:.4f}, 处理时间={proc_time:.2f}s {'✅' if success else '❌'}")
        except Exception as e:
            print(f"❌ 失败: {e}")

    # 统计
    print("\n" + "="*80)
    print("📊 性能对比")
    print("="*80)

    if results["remote"]:
        avg_remote = sum(results["remote"]) / len(results["remote"])
        print(f"\n🖥️  远程 CPU 平均 RTF: {avg_remote:.4f}")
        print(f"   处理速度: {1/avg_remote:.2f}x 实时")
        print(f"   日处理能力: {24/avg_remote:.2f} 小时")

    if results["local"]:
        avg_local = sum(results["local"]) / len(results["local"])
        print(f"\n🎮 本地 GPU 平均 RTF: {avg_local:.4f}")
        print(f"   处理速度: {1/avg_local:.2f}x 实时")
        print(f"   日处理能力: {24/avg_local:.2f} 小时")

    if results["remote"] and results["local"]:
        speedup = avg_remote / avg_local
        print(f"\n⚡ GPU 加速比: {speedup:.2f}x")

    # 清理
    import shutil
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
