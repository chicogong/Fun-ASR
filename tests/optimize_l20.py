#!/usr/bin/env python3
"""
测试 NVIDIA L20 的极限性能
找出最优批量大小和并发配置
"""

import requests
import time
import wave
import struct
import tempfile
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

REMOTE_URL = "http://asr-l20.trtc-ai.woa.com:8000"

def generate_wav(filepath, duration_sec=10, sample_rate=16000):
    num_samples = duration_sec * sample_rate
    with wave.open(filepath, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for _ in range(num_samples):
            wav.writeframes(struct.pack('<h', 0))
    return duration_sec

def test_batch(url, audio_paths, durations):
    total_dur = sum(durations)
    start = time.time()
    files = [("files", (os.path.basename(p), open(p, "rb"), "audio/wav")) for p in audio_paths]
    data = {"language": "zh"}
    try:
        r = requests.post(f"{url}/transcribe_batch", files=files, data=data, timeout=600)
        success = r.status_code == 200
    except Exception as e:
        success = False
    finally:
        for _, ft in files:
            ft[1].close()
    proc_time = time.time() - start
    rtf = proc_time / total_dur if total_dur > 0 else 0
    return rtf, proc_time, success

def test_concurrent_single(url, audio_path, duration, num_concurrent):
    """测试并发单文件请求"""
    def single_request():
        start = time.time()
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            data = {"language": "zh"}
            r = requests.post(f"{url}/transcribe", files=files, data=data, timeout=120)
        proc_time = time.time() - start
        return proc_time, r.status_code == 200

    start = time.time()
    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(single_request) for _ in range(num_concurrent)]
        results = [f.result() for f in as_completed(futures)]

    total_time = time.time() - start
    total_audio_duration = duration * num_concurrent
    rtf = total_time / total_audio_duration
    throughput = num_concurrent / total_time

    return rtf, total_time, throughput, all(r[1] for r in results)

def main():
    print("="*80)
    print("🚀 NVIDIA L20 性能优化测试")
    print("="*80)

    # 获取服务器信息
    info = requests.get(f"{REMOTE_URL}/info").json()
    print(f"\n📊 服务器配置:")
    print(f"  GPU: {info['gpu_name']}")
    print(f"  GPU 内存: {info['gpu_memory_used_gb']:.1f} / {info['gpu_memory_total_gb']:.1f} GB ({info['gpu_memory_used_gb']/info['gpu_memory_total_gb']*100:.1f}%)")
    print(f"  最大批量: {info['max_batch_size']}")

    # 生成测试文件
    temp_dir = tempfile.mkdtemp()
    print(f"\n🎵 生成测试音频（50 个 10 秒文件）...")

    test_files = []
    for i in range(50):
        path = os.path.join(temp_dir, f"test_{i:02d}.wav")
        generate_wav(path, 10)
        test_files.append(path)

    durations = [10] * 50

    # 测试不同批量大小
    print(f"\n{'='*80}")
    print("📦 测试不同批量大小（找出最优配置）")
    print("="*80)

    batch_results = {}
    batch_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for batch_size in batch_sizes:
        print(f"\n🔬 批量大小 {batch_size}:")
        batch_rtfs = []

        for run in range(3):
            batch_files = test_files[:batch_size]
            batch_durs = durations[:batch_size]

            rtf, proc, ok = test_batch(REMOTE_URL, batch_files, batch_durs)
            if ok:
                batch_rtfs.append(rtf)
                total_audio = sum(batch_durs)
                capacity = 24 / rtf if rtf > 0 else 0
                print(f"  Run {run+1}: RTF={rtf:.4f}, 处理={proc:.2f}s, 音频={total_audio}s, 日处理={capacity:.0f}h {'✅' if ok else '❌'}")
            else:
                print(f"  Run {run+1}: ❌ 失败")

            time.sleep(0.5)

        if batch_rtfs:
            avg_rtf = statistics.mean(batch_rtfs)
            min_rtf = min(batch_rtfs)
            batch_results[batch_size] = {
                'avg_rtf': avg_rtf,
                'min_rtf': min_rtf,
                'capacity': 24 / avg_rtf
            }
            print(f"  📊 平均 RTF: {avg_rtf:.4f}, 日处理: {24/avg_rtf:.0f} 小时")

    # 测试并发请求
    print(f"\n{'='*80}")
    print("⚡ 测试并发请求（单文件并发）")
    print("="*80)

    test_audio = test_files[0]
    concurrent_configs = [2, 4, 8, 16]

    concurrent_results = {}
    for num_concurrent in concurrent_configs:
        print(f"\n🔬 并发数 {num_concurrent}:")

        rtf, total_time, throughput, ok = test_concurrent_single(
            REMOTE_URL, test_audio, 10, num_concurrent
        )

        if ok:
            concurrent_results[num_concurrent] = {
                'rtf': rtf,
                'throughput': throughput,
                'capacity': 24 / rtf if rtf > 0 else 0
            }
            print(f"  总时间: {total_time:.2f}s")
            print(f"  RTF: {rtf:.4f}")
            print(f"  吞吐量: {throughput:.2f} 请求/秒")
            print(f"  日处理: {24/rtf:.0f} 小时 {'✅' if ok else '❌'}")

    # 分析结果
    print(f"\n{'='*80}")
    print("📊 性能分析报告")
    print("="*80)

    if batch_results:
        print(f"\n🏆 批量处理最优配置:")
        best_batch = max(batch_results.items(), key=lambda x: x[1]['capacity'])
        print(f"  最优批量大小: {best_batch[0]}")
        print(f"  RTF: {best_batch[1]['avg_rtf']:.4f}")
        print(f"  日处理能力: {best_batch[1]['capacity']:.0f} 小时")

        print(f"\n📈 批量大小性能对比:")
        print(f"  {'批量':<6} | {'平均RTF':<9} | {'最佳RTF':<9} | {'日处理(小时)':<12}")
        print(f"  {'-'*50}")
        for size, data in sorted(batch_results.items()):
            print(f"  {size:<6} | {data['avg_rtf']:<9.4f} | {data['min_rtf']:<9.4f} | {data['capacity']:<12.0f}")

    if concurrent_results:
        print(f"\n⚡ 并发处理最优配置:")
        best_concurrent = max(concurrent_results.items(), key=lambda x: x[1]['capacity'])
        print(f"  最优并发数: {best_concurrent[0]}")
        print(f"  RTF: {best_concurrent[1]['rtf']:.4f}")
        print(f"  吞吐量: {best_concurrent[1]['throughput']:.2f} 请求/秒")
        print(f"  日处理能力: {best_concurrent[1]['capacity']:.0f} 小时")

    # 推荐配置
    print(f"\n{'='*80}")
    print("💡 优化建议")
    print("="*80)

    if batch_results:
        best = max(batch_results.items(), key=lambda x: x[1]['capacity'])
        current_capacity = 669.76  # 当前批量5的性能

        print(f"\n1️⃣ 增加批量大小:")
        print(f"   当前配置: 批量大小 5, 日处理 670 小时")
        print(f"   推荐配置: 批量大小 {best[0]}, 日处理 {best[1]['capacity']:.0f} 小时")
        print(f"   性能提升: {best[1]['capacity']/current_capacity:.2f}x 倍")

    print(f"\n2️⃣ 其他优化方向:")
    print(f"   - GPU 利用率仅 4.5%，可以:")
    print(f"     • 运行多个 uvicorn worker")
    print(f"     • 使用模型并行或流水线并行")
    print(f"     • 实现真正的批量推理（而非顺序处理）")
    print(f"   - 模型优化:")
    print(f"     • 使用 FP16 精度（可能 2x 加速）")
    print(f"     • TensorRT 优化")
    print(f"   - 系统优化:")
    print(f"     • 异步处理队列")
    print(f"     • 动态批处理")

    # 理论极限估算
    print(f"\n3️⃣ 理论极限估算:")
    if batch_results:
        best_rtf = min(data['min_rtf'] for data in batch_results.values())
        gpu_util = info['gpu_memory_used_gb'] / info['gpu_memory_total_gb']

        # 假设可以充分利用 GPU（达到 80% 利用率）
        theoretical_speedup = 0.8 / gpu_util
        theoretical_capacity = (24 / best_rtf) * theoretical_speedup

        print(f"   当前 GPU 利用率: {gpu_util*100:.1f}%")
        print(f"   假设优化到 80% 利用率:")
        print(f"   理论最大日处理: {theoretical_capacity:.0f} 小时")
        print(f"   相比当前提升: {theoretical_capacity/current_capacity:.1f}x 倍")

    # 清理
    import shutil
    shutil.rmtree(temp_dir)

    print(f"\n✅ 测试完成!")
    print("="*80)

if __name__ == "__main__":
    main()
