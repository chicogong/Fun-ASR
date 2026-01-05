#!/usr/bin/env python3
"""测试远程服务器优化后的性能（MAX_BATCH_SIZE=50）"""

import requests
import time
import wave
import struct
import tempfile
import os
import statistics

REMOTE_URL = os.environ.get("REMOTE_URL", "http://localhost:8000")

def generate_wav(filepath, duration_sec=10, sample_rate=16000):
    num_samples = duration_sec * sample_rate
    with wave.open(filepath, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for _ in range(num_samples):
            wav.writeframes(struct.pack('<h', 0))

def test_batch(url, files, durations):
    total_dur = sum(durations)
    start = time.time()

    file_handles = [("files", (os.path.basename(p), open(p, "rb"), "audio/wav")) for p in files]
    data = {"language": "zh"}

    try:
        r = requests.post(f"{url}/transcribe_batch", files=file_handles, data=data, timeout=600)
        success = r.status_code == 200
        if not success:
            error_msg = r.text[:200]
        else:
            error_msg = None
    except Exception as e:
        success = False
        error_msg = str(e)[:200]
    finally:
        for _, fh in file_handles:
            fh[1].close()

    proc_time = time.time() - start
    rtf = proc_time / total_dur if total_dur > 0 else 0

    return rtf, proc_time, success, error_msg

def main():
    print("="*80)
    print("🚀 远程服务器优化性能测试（MAX_BATCH_SIZE=50）")
    print("="*80)

    # 检查服务配置
    try:
        info = requests.get(f"{REMOTE_URL}/info").json()
        print(f"\n📊 服务器配置:")
        print(f"  GPU: {info['gpu_name']}")
        print(f"  设备: {info['device']}")
        print(f"  最大批量: {info['max_batch_size']}")
        print(f"  GPU 内存: {info['gpu_memory_used_gb']:.1f} / {info['gpu_memory_total_gb']:.1f} GB")

        if info['max_batch_size'] != 50:
            print(f"\n⚠️  警告: max_batch_size = {info['max_batch_size']}, 期望 50")
    except Exception as e:
        print(f"\n❌ 无法连接到服务器: {e}")
        return

    # 生成测试文件
    temp_dir = tempfile.mkdtemp()
    print(f"\n🎵 生成 50 个测试音频文件（10秒/个）...")

    files = []
    for i in range(50):
        path = os.path.join(temp_dir, f"test_{i:02d}.wav")
        generate_wav(path, 10)
        files.append(path)

    durations = [10] * 50
    print(f"  ✅ 生成完成，总时长: {sum(durations)} 秒")

    # 测试不同批量大小
    print(f"\n{'='*80}")
    print("📦 批量大小测试（关键：测试 >20 的批量）")
    print("="*80)

    # 对比：旧版本最大20，新版本最大50
    batch_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    results = {}

    for size in batch_sizes:
        print(f"\n🔬 批量大小 {size}:")
        batch_rtfs = []

        for run in range(2):
            batch_files = files[:size]
            batch_durs = durations[:size]

            rtf, proc, ok, err = test_batch(REMOTE_URL, batch_files, batch_durs)

            if ok:
                batch_rtfs.append(rtf)
                capacity = 24 / rtf if rtf > 0 else 0
                print(f"  Run {run+1}: RTF={rtf:.4f}, 处理={proc:.2f}s, 日处理={capacity:.0f}h ✅")
            else:
                print(f"  Run {run+1}: ❌ 失败 - {err}")

            time.sleep(0.5)

        if batch_rtfs:
            avg_rtf = statistics.mean(batch_rtfs)
            min_rtf = min(batch_rtfs)
            results[size] = {
                'avg_rtf': avg_rtf,
                'min_rtf': min_rtf,
                'capacity': 24 / avg_rtf
            }
            print(f"  📊 平均: RTF={avg_rtf:.4f}, 日处理={24/avg_rtf:.0f}h")

    # 性能分析
    print(f"\n{'='*80}")
    print("📊 性能分析报告")
    print("="*80)

    if results:
        # 找出最优批量
        best_size = min(results.keys(), key=lambda x: results[x]['avg_rtf'])
        best_data = results[best_size]

        print(f"\n🏆 最优配置:")
        print(f"  批量大小: {best_size}")
        print(f"  平均 RTF: {best_data['avg_rtf']:.4f}")
        print(f"  最佳 RTF: {best_data['min_rtf']:.4f}")
        print(f"  日处理: {best_data['capacity']:.0f} 小时")

        # 性能对比表
        print(f"\n📈 批量大小性能对比:")
        print(f"  {'批量':>5} | {'平均RTF':>9} | {'最佳RTF':>9} | {'日处理(h)':>10} | {'状态':>6}")
        print(f"  {'-'*58}")

        for size in sorted(results.keys()):
            data = results[size]
            status = "✅" if size <= 20 else "🆕"  # 标记新功能
            print(f"  {size:>5} | {data['avg_rtf']:>9.4f} | {data['min_rtf']:>9.4f} | {data['capacity']:>10.0f} | {status:>6}")

        # 与旧版本对比
        print(f"\n💡 性能提升分析:")

        if 5 in results and best_size in results:
            baseline = results[5]
            optimized = results[best_size]

            rtf_improvement = baseline['avg_rtf'] / optimized['avg_rtf']
            capacity_improvement = optimized['capacity'] / baseline['capacity']

            print(f"  基准（批量5）: RTF={baseline['avg_rtf']:.4f}, 日处理={baseline['capacity']:.0f}h")
            print(f"  最优（批量{best_size}）: RTF={optimized['avg_rtf']:.4f}, 日处理={optimized['capacity']:.0f}h")
            print(f"  性能提升: {rtf_improvement:.2f}x 倍")
            print(f"  容量提升: {capacity_improvement:.2f}x 倍")

        # 验证新功能
        large_batches = [s for s in results.keys() if s > 20]
        if large_batches:
            print(f"\n✅ 新功能验证: 支持批量 >20")
            print(f"  成功测试的批量: {', '.join(map(str, sorted(large_batches)))}")

            best_large = min(large_batches, key=lambda x: results[x]['avg_rtf'])
            print(f"  大批量最优: 批量 {best_large}, RTF={results[best_large]['avg_rtf']:.4f}")
        else:
            print(f"\n⚠️  警告: 没有批量 >20 的测试成功")

    # GPU 利用率分析
    try:
        final_info = requests.get(f"{REMOTE_URL}/info").json()
        gpu_util = final_info['gpu_memory_used_gb'] / final_info['gpu_memory_total_gb'] * 100

        print(f"\n🎮 GPU 利用率:")
        print(f"  当前使用: {final_info['gpu_memory_used_gb']:.1f} GB")
        print(f"  总容量: {final_info['gpu_memory_total_gb']:.1f} GB")
        print(f"  利用率: {gpu_util:.1f}%")

        if gpu_util < 20:
            potential = final_info['gpu_memory_total_gb'] / final_info['gpu_memory_used_gb']
            print(f"  💡 优化潜力: 理论上可提升 {potential:.1f}x 倍（通过增加并发/优化）")
    except:
        pass

    # 清理
    import shutil
    shutil.rmtree(temp_dir)

    print(f"\n✅ 测试完成!")
    print("="*80)

if __name__ == "__main__":
    main()
