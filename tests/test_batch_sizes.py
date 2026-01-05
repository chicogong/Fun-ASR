#!/usr/bin/env python3
"""测试不同批量大小的性能"""

import requests
import time
import wave
import struct
import tempfile
import os
import statistics

URL = "http://localhost:8000"

def generate_wav(filepath, duration_sec=10, sample_rate=16000):
    num_samples = duration_sec * sample_rate
    with wave.open(filepath, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for _ in range(num_samples):
            wav.writeframes(struct.pack('<h', 0))

def test_batch(files, durations):
    total_dur = sum(durations)
    start = time.time()

    file_handles = [("files", (os.path.basename(p), open(p, "rb"), "audio/wav")) for p in files]
    data = {"language": "zh"}

    try:
        r = requests.post(f"{URL}/transcribe_batch", files=file_handles, data=data, timeout=300)
        success = r.status_code == 200
        if not success:
            print(f"    Error: {r.status_code} - {r.text[:100]}")
    except Exception as e:
        success = False
        print(f"    Exception: {e}")
    finally:
        for _, fh in file_handles:
            fh[1].close()

    proc_time = time.time() - start
    rtf = proc_time / total_dur if total_dur > 0 else 0

    return rtf, proc_time, success

def main():
    print("="*60)
    print("🧪 批量大小测试")
    print("="*60)

    # 检查服务
    try:
        info = requests.get(f"{URL}/info").json()
        print(f"\n当前配置:")
        print(f"  max_batch_size: {info['max_batch_size']}")
        print(f"  device: {info['device']}")
    except:
        print("❌ 服务未运行")
        return

    # 生成测试文件
    temp_dir = tempfile.mkdtemp()
    print(f"\n📁 生成 30 个测试文件...")

    files = []
    for i in range(30):
        path = os.path.join(temp_dir, f"test_{i:02d}.wav")
        generate_wav(path, 10)
        files.append(path)

    durations = [10] * 30

    # 测试不同批量
    print(f"\n{'='*60}")
    print("📊 测试结果")
    print("="*60)

    batch_sizes = [5, 10, 15, 20, 25, 30]
    results = {}

    for size in batch_sizes:
        print(f"\n批量 {size}:")
        rtfs = []

        for run in range(2):
            batch_files = files[:size]
            batch_durs = durations[:size]

            rtf, proc, ok = test_batch(batch_files, batch_durs)

            if ok:
                rtfs.append(rtf)
                capacity = 24 / rtf if rtf > 0 else 0
                print(f"  Run {run+1}: RTF={rtf:.4f}, 处理={proc:.2f}s, 日处理={capacity:.0f}h ✅")
            else:
                print(f"  Run {run+1}: ❌ 失败")

        if rtfs:
            avg_rtf = statistics.mean(rtfs)
            results[size] = avg_rtf
            print(f"  平均: RTF={avg_rtf:.4f}, 日处理={24/avg_rtf:.0f}h")

    # 总结
    if results:
        print(f"\n{'='*60}")
        print("📈 性能总结")
        print("="*60)

        best_size = min(results.keys(), key=lambda x: results[x])
        print(f"\n最优批量: {best_size}")
        print(f"最佳 RTF: {results[best_size]:.4f}")
        print(f"日处理: {24/results[best_size]:.0f} 小时")

        print(f"\n批量大小对比:")
        for size, rtf in sorted(results.items()):
            print(f"  批量 {size:2d}: RTF={rtf:.4f}, 日处理={24/rtf:5.0f}h")

    # 清理
    import shutil
    shutil.rmtree(temp_dir)

    print(f"\n✅ 测试完成!")

if __name__ == "__main__":
    main()
