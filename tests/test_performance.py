#!/usr/bin/env python3
"""
Fun-ASR 性能测试脚本

测试内容:
1. 串行 vs 批量处理对比
2. 不同 batch size 吞吐量测试
3. 最优 batch size 查找

使用方法:
    python tests/test_performance.py [--api-url URL] [--audio-dir DIR]

测试结果 (Tesla T4 15GB):
    - 最优 batch size: 20
    - 最高吞吐: ~10 files/s
    - 实时率: ~50-60x
"""

import argparse
import time
import wave
from pathlib import Path
from typing import List
import requests


def get_audio_files(directory: str, max_files: int = 100) -> List[Path]:
    """递归获取音频文件"""
    audio_dir = Path(directory)
    extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus'}
    files = []
    for f in audio_dir.rglob('*'):
        if f.is_file() and f.suffix.lower() in extensions:
            files.append(f)
            if len(files) >= max_files:
                break
    return files


def get_audio_duration(files: List[Path]) -> float:
    """计算音频总时长(秒)"""
    total = 0
    for f in files:
        try:
            with wave.open(str(f), 'rb') as w:
                total += w.getnframes() / w.getframerate()
        except:
            total += 7.5  # 估计值
    return total


def test_serial(api_url: str, files: List[Path], num_files: int = 10) -> dict:
    """测试串行处理 (逐个文件)"""
    test_files = files[:num_files]
    
    start = time.time()
    for f in test_files:
        with open(f, 'rb') as audio:
            requests.post(
                f"{api_url}/transcribe",
                files={"file": (f.name, audio, "audio/wav")},
                data={"model": "nano"},
                timeout=60
            )
    elapsed = time.time() - start
    
    # 按比例估算
    estimated_time = elapsed * (len(files) / num_files)
    throughput = len(files) / estimated_time
    
    return {
        "mode": "serial",
        "files": len(files),
        "time": estimated_time,
        "throughput": throughput,
    }


def test_batch(api_url: str, files: List[Path], batch_size: int) -> dict:
    """测试批量处理"""
    batch_files = files[:batch_size]
    multipart = [('files', (f.name, open(f, 'rb'), 'audio/wav')) for f in batch_files]
    
    try:
        start = time.time()
        resp = requests.post(
            f"{api_url}/transcribe_batch",
            files=multipart,
            data={"model": "nano"},
            timeout=120
        )
        elapsed = time.time() - start
        
        if resp.status_code == 200:
            return {
                "mode": "batch",
                "batch_size": batch_size,
                "time": elapsed,
                "throughput": batch_size / elapsed,
                "success": True,
            }
        return {"success": False, "error": resp.text[:100]}
    except Exception as e:
        return {"success": False, "error": str(e)[:100]}
    finally:
        for _, (_, f, _) in multipart:
            f.close()


def find_optimal_batch(api_url: str, files: List[Path]) -> int:
    """查找最优 batch size"""
    print("\n查找最优 batch size...")
    print("-" * 45)
    print("Batch Size | 吞吐 (files/s) | 耗时 (s)")
    print("-" * 45)
    
    best_bs, best_tp = 0, 0
    
    for bs in [8, 12, 16, 18, 20, 22, 24, 28, 32]:
        if bs > len(files):
            break
        
        result = test_batch(api_url, files, bs)
        if result.get("success", False):
            tp = result["throughput"]
            print(f"    {bs:3d}    |     {tp:.2f}       |   {result['time']:.2f}")
            if tp > best_tp:
                best_tp = tp
                best_bs = bs
        else:
            print(f"    {bs:3d}    |     失败        |   -")
        
        time.sleep(0.5)
    
    print("-" * 45)
    print(f"最优: batch={best_bs}, 吞吐={best_tp:.2f} files/s")
    return best_bs


def run_comparison(api_url: str, files: List[Path]):
    """运行串行 vs 批量对比测试"""
    duration = get_audio_duration(files)
    
    print("=" * 50)
    print("Fun-ASR 性能对比测试")
    print("=" * 50)
    print(f"音频文件数: {len(files)}")
    print(f"音频总时长: {duration:.1f}秒 ({duration/60:.1f}分钟)")
    
    # 串行测试
    print("\n测试串行处理...")
    serial = test_serial(api_url, files, num_files=10)
    print(f"  吞吐: {serial['throughput']:.2f} files/s")
    print(f"  实时率: {duration/serial['time']:.1f}x")
    
    # 批量测试
    print("\n测试批量处理 (batch=20)...")
    batch = test_batch(api_url, files, min(20, len(files)))
    if batch.get("success"):
        print(f"  吞吐: {batch['throughput']:.2f} files/s")
        batch_time_full = (len(files) / batch['batch_size']) * batch['time']
        print(f"  实时率: {duration/batch_time_full:.1f}x")
        
        # 对比
        speedup = serial['time'] / batch_time_full
        print(f"\n批量处理比串行快 {speedup:.1f}x")
    else:
        print(f"  失败: {batch.get('error')}")


def main():
    parser = argparse.ArgumentParser(description="Fun-ASR 性能测试")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API 地址")
    parser.add_argument("--audio-dir", default="/data/workspace/1", help="音频文件目录")
    parser.add_argument("--find-optimal", action="store_true", help="查找最优 batch size")
    args = parser.parse_args()
    
    # 检查服务
    try:
        resp = requests.get(f"{args.api_url}/health", timeout=5)
        if resp.status_code != 200:
            print(f"服务不可用: {args.api_url}")
            return
    except:
        print(f"无法连接服务: {args.api_url}")
        return
    
    # 加载音频文件
    files = get_audio_files(args.audio_dir)
    if not files:
        print(f"未找到音频文件: {args.audio_dir}")
        return
    
    if args.find_optimal:
        find_optimal_batch(args.api_url, files)
    else:
        run_comparison(args.api_url, files)


if __name__ == "__main__":
    main()
