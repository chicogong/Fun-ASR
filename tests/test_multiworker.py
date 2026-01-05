#!/usr/bin/env python3
"""测试多 Worker 性能提升"""

import requests
import time
import wave
import struct
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://localhost:8000"

def generate_wav(filepath, duration_sec=10, sample_rate=16000):
    num_samples = duration_sec * sample_rate
    with wave.open(filepath, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for _ in range(num_samples):
            wav.writeframes(struct.pack('<h', 0))

def single_request(audio_path, duration):
    """单个请求"""
    start = time.time()
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
        data = {"language": "zh"}
        r = requests.post(f"{URL}/transcribe", files=files, data=data, timeout=120)
    proc_time = time.time() - start
    rtf = proc_time / duration
    return rtf, proc_time, r.status_code == 200

def concurrent_test(audio_path, duration, num_requests):
    """并发测试"""
    def make_request():
        return single_request(audio_path, duration)

    start_total = time.time()

    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        results = [f.result() for f in as_completed(futures)]

    total_time = time.time() - start_total
    total_audio_duration = duration * num_requests
    overall_rtf = total_time / total_audio_duration
    throughput = num_requests / total_time

    successful = sum(1 for r in results if r[2])

    return {
        'num_requests': num_requests,
        'total_time': total_time,
        'total_audio': total_audio_duration,
        'overall_rtf': overall_rtf,
        'throughput': throughput,
        'successful': successful,
        'avg_individual_rtf': sum(r[0] for r in results) / len(results)
    }

def main():
    print("="*80)
    print("🚀 多 Worker 性能测试")
    print("="*80)

    # 检查服务
    info = requests.get(f"{URL}/info").json()
    print(f"\n📊 服务配置:")
    print(f"  Workers: {info.get('num_workers', 1)}")
    print(f"  Device: {info['device']}")
    print(f"  GPU: {info.get('gpu_name', 'N/A')}")

    # 生成测试文件
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "test_10s.wav")
    generate_wav(audio_path, 10)
    duration = 10

    print(f"\n🎵 测试音频: 10秒")

    # 测试不同并发数
    print(f"\n{'='*80}")
    print("📊 并发性能测试")
    print("="*80)

    concurrent_levels = [1, 2, 4, 8, 12, 16, 20]

    print(f"\n{'并发数':>6} | {'总时间(s)':>10} | {'整体RTF':>10} | {'吞吐量':>12} | {'日处理(h)':>12}")
    print("-" * 80)

    best_throughput = 0
    best_config = None

    for num in concurrent_levels:
        result = concurrent_test(audio_path, duration, num)

        daily_capacity = 24 / result['overall_rtf'] if result['overall_rtf'] > 0 else 0

        print(f"{num:>6} | {result['total_time']:>10.2f} | {result['overall_rtf']:>10.4f} | "
              f"{result['throughput']:>12.2f} | {daily_capacity:>12.0f}")

        if result['throughput'] > best_throughput:
            best_throughput = result['throughput']
            best_config = (num, result)

    # 总结
    print(f"\n{'='*80}")
    print("🏆 最优配置")
    print("="*80)

    if best_config:
        num, result = best_config
        daily_capacity = 24 / result['overall_rtf']

        print(f"\n最佳并发数: {num}")
        print(f"整体 RTF: {result['overall_rtf']:.4f}")
        print(f"吞吐量: {result['throughput']:.2f} 请求/秒")
        print(f"日处理能力: {daily_capacity:.0f} 小时")

        # 与单 Worker 对比（假设单Worker RTF=0.06）
        single_worker_capacity = 24 / 0.06  # 400 小时/天
        improvement = daily_capacity / single_worker_capacity

        print(f"\n💡 性能提升:")
        print(f"  单 Worker 估计: ~400 小时/天")
        print(f"  多 Worker 实际: {daily_capacity:.0f} 小时/天")
        print(f"  提升倍数: {improvement:.2f}x")

    # 清理
    import shutil
    shutil.rmtree(temp_dir)

    print(f"\n✅ 测试完成!")
    print("="*80)

if __name__ == "__main__":
    main()
