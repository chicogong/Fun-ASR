#!/usr/bin/env python3
"""测试远程服务器 GPU 模式性能"""

import requests
import time
import wave
import struct
import tempfile
import os

REMOTE_URL = os.environ.get("REMOTE_URL", "http://localhost:8000")

def generate_wav(filepath, duration_sec=10, sample_rate=16000):
    num_samples = duration_sec * sample_rate
    with wave.open(filepath, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for _ in range(num_samples):
            wav.writeframes(struct.pack('<h', 0))
    return duration_sec

def test_single(url, audio_path, duration):
    start = time.time()
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
        data = {"language": "zh"}
        r = requests.post(f"{url}/transcribe", files=files, data=data, timeout=120)
    proc_time = time.time() - start
    rtf = proc_time / duration
    return rtf, proc_time, r.status_code == 200

def test_batch(url, audio_paths, durations):
    total_dur = sum(durations)
    start = time.time()
    files = [("files", (os.path.basename(p), open(p, "rb"), "audio/wav")) for p in audio_paths]
    data = {"language": "zh"}
    r = requests.post(f"{url}/transcribe_batch", files=files, data=data, timeout=300)
    for _, ft in files:
        ft[1].close()
    proc_time = time.time() - start
    rtf = proc_time / total_dur
    return rtf, proc_time, r.status_code == 200

def main():
    print("="*80)
    print("🎮 远程服务器 GPU 模式性能测试")
    print("="*80)

    # 获取服务器信息
    info = requests.get(f"{REMOTE_URL}/info").json()
    print(f"\n📊 服务器配置:")
    print(f"  设备: {info['device']}")
    print(f"  GPU: {info['gpu_name']}")
    print(f"  GPU 内存: {info['gpu_memory_used_gb']:.1f} / {info['gpu_memory_total_gb']:.1f} GB")

    # 生成测试文件
    temp_dir = tempfile.mkdtemp()
    test_files = []
    durations = [5, 10, 20, 30]

    print(f"\n🎵 生成测试音频...")
    for dur in durations:
        path = os.path.join(temp_dir, f"test_{dur}s.wav")
        generate_wav(path, dur)
        test_files.append((path, dur))

    # 单文件测试
    print(f"\n{'='*80}")
    print("📄 单文件测试")
    print("="*80)

    rtfs = []
    for path, dur in test_files:
        rtf, proc, ok = test_single(REMOTE_URL, path, dur)
        rtfs.append(rtf)
        print(f"  {dur:2d}s 音频: RTF={rtf:.4f}, 处理={proc:.2f}s {'✅' if ok else '❌'}")

    avg_rtf = sum(rtfs) / len(rtfs)
    print(f"\n  平均 RTF: {avg_rtf:.4f}")
    print(f"  处理速度: {1/avg_rtf:.2f}x 实时")
    print(f"  日处理能力: {24/avg_rtf:.2f} 小时")

    # 批量测试
    print(f"\n{'='*80}")
    print("📦 批量测试 (批量大小 5)")
    print("="*80)

    # 生成 5 个 10 秒的文件
    batch_files = []
    batch_durs = []
    for i in range(5):
        path = os.path.join(temp_dir, f"batch_{i}.wav")
        generate_wav(path, 10)
        batch_files.append(path)
        batch_durs.append(10)

    batch_rtfs = []
    for run in range(3):
        rtf, proc, ok = test_batch(REMOTE_URL, batch_files, batch_durs)
        batch_rtfs.append(rtf)
        print(f"  Run {run+1}: RTF={rtf:.4f}, 处理={proc:.2f}s, 总音频={sum(batch_durs)}s {'✅' if ok else '❌'}")

    avg_batch_rtf = sum(batch_rtfs) / len(batch_rtfs)
    print(f"\n  平均 RTF: {avg_batch_rtf:.4f}")
    print(f"  处理速度: {1/avg_batch_rtf:.2f}x 实时")
    print(f"  日处理能力: {24/avg_batch_rtf:.2f} 小时")

    # 对比
    print(f"\n{'='*80}")
    print("📊 性能对比总结")
    print("="*80)
    print(f"\n之前 CPU 模式:")
    print(f"  RTF: 0.3083")
    print(f"  日处理: 77.84 小时")
    print(f"\n现在 GPU 模式 (NVIDIA L20):")
    print(f"  RTF: {avg_rtf:.4f} (单文件)")
    print(f"  RTF: {avg_batch_rtf:.4f} (批量大小5)")
    print(f"  日处理: {24/avg_batch_rtf:.2f} 小时")
    print(f"\n🚀 性能提升: {0.3083/avg_batch_rtf:.2f}x 倍")

    # 清理
    import shutil
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
