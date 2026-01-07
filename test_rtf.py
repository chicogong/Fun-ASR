#!/usr/bin/env python3
"""
RTF (Real-Time Factor) 测试脚本
测试批量推理性能并计算每天可处理的音频小时数
"""

import time
import requests
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np


def get_audio_duration(audio_path):
    """获取音频时长（秒）"""
    try:
        data, samplerate = sf.read(audio_path)
        duration = len(data) / samplerate
        return duration
    except Exception as e:
        print(f"⚠️  无法读取音频文件 {audio_path}: {e}")
        return None


def test_single_file(server_url, audio_path, language="auto"):
    """测试单文件推理"""
    print(f"\n{'='*60}")
    print("📊 单文件推理测试")
    print(f"{'='*60}")

    audio_duration = get_audio_duration(audio_path)
    if audio_duration is None:
        return None

    print(f"📁 文件: {audio_path}")
    print(f"⏱️  音频时长: {audio_duration:.2f}s")

    with open(audio_path, 'rb') as f:
        start_time = time.time()
        response = requests.post(
            f"{server_url}/transcribe",
            files={'file': f},
            data={'language': language}
        )
        process_time = time.time() - start_time

    if response.status_code != 200:
        print(f"❌ 请求失败: {response.status_code}")
        return None

    result = response.json()
    rtf = process_time / audio_duration

    print(f"🚀 处理时间: {process_time:.2f}s")
    print(f"📈 RTF: {rtf:.4f} ({1/rtf:.1f}x 实时速度)")
    print(f"📝 转写结果: {result['results'][0]['text'][:100]}...")

    return {
        'audio_duration': audio_duration,
        'process_time': process_time,
        'rtf': rtf
    }


def test_batch(server_url, audio_path, batch_size, language="auto"):
    """测试批量推理"""
    print(f"\n{'='*60}")
    print(f"📊 批量推理测试 (Batch Size = {batch_size})")
    print(f"{'='*60}")

    audio_duration = get_audio_duration(audio_path)
    if audio_duration is None:
        return None

    total_audio_duration = audio_duration * batch_size

    print(f"📁 文件: {audio_path}")
    print(f"⏱️  单个音频时长: {audio_duration:.2f}s")
    print(f"⏱️  总音频时长: {total_audio_duration:.2f}s ({total_audio_duration/3600:.2f}h)")

    # 准备批量文件
    files = []
    for i in range(batch_size):
        files.append(('files', open(audio_path, 'rb')))

    try:
        start_time = time.time()
        response = requests.post(
            f"{server_url}/transcribe_batch",
            files=files,
            data={'language': language}
        )
        process_time = time.time() - start_time
    finally:
        # 关闭文件句柄
        for _, f in files:
            f.close()

    if response.status_code != 200:
        print(f"❌ 请求失败: {response.status_code}")
        return None

    result = response.json()
    rtf = process_time / total_audio_duration
    avg_time_per_file = process_time / batch_size

    print(f"🚀 总处理时间: {process_time:.2f}s")
    print(f"⚡ 平均每文件: {avg_time_per_file:.2f}s")
    print(f"📈 Batch RTF: {rtf:.4f} ({1/rtf:.1f}x 实时速度)")

    return {
        'batch_size': batch_size,
        'audio_duration': audio_duration,
        'total_audio_duration': total_audio_duration,
        'process_time': process_time,
        'avg_time_per_file': avg_time_per_file,
        'rtf': rtf
    }


def calculate_throughput(rtf, batch_size=1):
    """计算每天可处理的音频小时数"""
    print(f"\n{'='*60}")
    print("📊 吞吐量计算")
    print(f"{'='*60}")

    seconds_per_day = 24 * 3600  # 86400秒

    # 如果RTF=0.03，意味着处理1秒音频需要0.03秒
    # 所以1秒可以处理 1/0.03 = 33.33秒的音频
    audio_seconds_per_second = 1.0 / rtf

    # 每天可处理的音频秒数
    audio_seconds_per_day = audio_seconds_per_second * seconds_per_day

    # 转换为小时
    audio_hours_per_day = audio_seconds_per_day / 3600

    print(f"🎯 RTF: {rtf:.4f}")
    print(f"⚡ 实时速度倍数: {audio_seconds_per_second:.1f}x")
    print(f"📊 Batch Size: {batch_size}")
    print(f"\n{'─'*60}")
    print(f"📈 每秒可处理: {audio_seconds_per_second:.1f}秒音频")
    print(f"📈 每分钟可处理: {audio_seconds_per_second*60:.0f}秒音频 ({audio_seconds_per_second:.1f}分钟)")
    print(f"📈 每小时可处理: {audio_seconds_per_second*3600/3600:.1f}小时音频")
    print(f"\n{'─'*60}")
    print(f"🎉 每天可处理: {audio_hours_per_day:.0f} 小时音频")
    print(f"🎉 每天可处理: {audio_hours_per_day/24:.0f} 天音频")
    print(f"{'─'*60}")

    # 如果是批量模式，计算并行优势
    if batch_size > 1:
        print(f"\n💡 批量优势:")
        print(f"   使用 batch_size={batch_size} 相比单个处理")
        print(f"   可以充分利用GPU并行计算能力")

    return audio_hours_per_day


def run_comprehensive_test(server_url, audio_path, batch_sizes, language="auto"):
    """运行完整的性能测试"""
    print(f"\n{'='*60}")
    print("🚀 Fun-ASR MLT 性能测试")
    print(f"{'='*60}")
    print(f"🌐 服务器: {server_url}")
    print(f"📁 测试文件: {audio_path}")
    print(f"🌍 语言: {language}")
    print(f"{'='*60}")

    # 检查服务器健康
    try:
        response = requests.get(f"{server_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ 服务器状态: {health['status']}")
            print(f"📦 模型: {health['model']}")
            print(f"🔧 设备: {health['device']}")
        else:
            print(f"⚠️  服务器健康检查失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
        return

    results = []

    # 测试单文件
    single_result = test_single_file(server_url, audio_path, language)
    if single_result:
        results.append(('单文件', 1, single_result))

    # 测试不同批量大小
    for batch_size in batch_sizes:
        batch_result = test_batch(server_url, audio_path, batch_size, language)
        if batch_result:
            results.append((f'Batch-{batch_size}', batch_size, batch_result))

    # 汇总结果
    print(f"\n{'='*60}")
    print("📊 性能汇总")
    print(f"{'='*60}")
    print(f"{'模式':<15} {'RTF':<10} {'实时倍数':<12} {'每天可处理(小时)':<20}")
    print(f"{'─'*60}")

    best_throughput = 0
    best_config = None

    for name, batch_size, result in results:
        rtf = result['rtf']
        speedup = 1.0 / rtf
        throughput = calculate_throughput(rtf, batch_size)

        print(f"{name:<15} {rtf:<10.4f} {speedup:<12.1f}x {throughput:<20.0f}")

        if throughput > best_throughput:
            best_throughput = throughput
            best_config = (name, batch_size, rtf)

    print(f"{'─'*60}")

    if best_config:
        print(f"\n🏆 最优配置: {best_config[0]}")
        print(f"   RTF: {best_config[2]:.4f}")
        print(f"   每天可处理: {best_throughput:.0f} 小时音频")

        # 计算加速比
        if len(results) > 1:
            single_rtf = results[0][2]['rtf']
            best_rtf = best_config[2]
            speedup_ratio = single_rtf / best_rtf
            print(f"   相比单文件加速: {speedup_ratio:.2f}x")

    # 计算详细吞吐量
    if best_config:
        print(f"\n{'='*60}")
        calculate_throughput(best_config[2], best_config[1])


def main():
    parser = argparse.ArgumentParser(description="Fun-ASR RTF性能测试")
    parser.add_argument('--server', default='http://localhost:8088',
                        help='服务器地址 (默认: http://localhost:8088)')
    parser.add_argument('--audio', required=True,
                        help='测试音频文件路径')
    parser.add_argument('--language', default='auto',
                        help='语言代码 (默认: auto)')
    parser.add_argument('--batch-sizes', default='1,3,6,10',
                        help='要测试的批量大小，逗号分隔 (默认: 1,3,6,10)')

    args = parser.parse_args()

    # 解析批量大小
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]

    # 检查文件是否存在
    if not Path(args.audio).exists():
        print(f"❌ 音频文件不存在: {args.audio}")
        return

    # 运行测试
    run_comprehensive_test(args.server, args.audio, batch_sizes, args.language)


if __name__ == '__main__':
    main()
