#!/usr/bin/env python3
"""
Fun-ASR WER/CER 测试脚本

使用 AISHELL-1 测试集评估模型准确率

使用方法:
    python tests/test_wer.py [--num-samples 100] [--api-url URL]
"""

import os
import argparse
import tempfile
import requests
import soundfile as sf
from jiwer import wer, cer

# 使用 HF 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def test_aishell(api_url: str, num_samples: int = 100):
    """使用 AISHELL-1 测试集"""
    from datasets import load_dataset
    
    print("加载 AISHELL-1 测试集...")
    dataset = load_dataset(
        "carlot/AIShell",
        split="test",
        streaming=True
    )
    
    references = []
    hypotheses = []
    
    print(f"\n测试 {num_samples} 条数据...")
    print("-" * 50)
    
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        
        audio = sample["audio"]
        reference = sample["text"]
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio["array"], audio["sampling_rate"])
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as audio_file:
                resp = requests.post(
                    f"{api_url}/transcribe",
                    files={"file": ("test.wav", audio_file, "audio/wav")},
                    data={"model": "nano", "language": "中文"},
                    timeout=30
                )
            
            if resp.status_code == 200:
                hypothesis = resp.json()["text"]
                references.append(reference)
                hypotheses.append(hypothesis)
                
                if (i + 1) % 20 == 0:
                    print(f"  已处理: {i+1}/{num_samples}")
        except Exception as e:
            print(f"  样本 {i} 失败: {str(e)[:50]}")
        finally:
            os.unlink(temp_path)
    
    # 计算结果
    print("\n" + "=" * 50)
    print("AISHELL-1 测试结果")
    print("=" * 50)
    
    error_cer = cer(references, hypotheses)
    error_wer = wer(references, hypotheses)
    
    print(f"  测试样本数: {len(references)}")
    print(f"  CER (字错误率): {error_cer*100:.2f}%")
    print(f"  WER (词错误率): {error_wer*100:.2f}%")
    print(f"  准确率: {(1-error_cer)*100:.2f}%")
    
    # 展示示例
    print("\n示例对比:")
    print("-" * 50)
    for i in range(min(5, len(references))):
        print(f"参考: {references[i]}")
        print(f"识别: {hypotheses[i]}")
        print()
    
    return error_cer, error_wer


def test_wenetspeech(api_url: str, num_samples: int = 100):
    """使用 WenetSpeech 测试集"""
    from datasets import load_dataset
    
    print("加载 WenetSpeech 测试集...")
    dataset = load_dataset(
        "wenet-e2e/wenetspeech",
        "test_meeting",
        split="test",
        streaming=True
    )
    
    references = []
    hypotheses = []
    
    print(f"\n测试 {num_samples} 条数据...")
    
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        
        audio = sample["audio"]
        reference = sample["text"]
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio["array"], audio["sampling_rate"])
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as audio_file:
                resp = requests.post(
                    f"{api_url}/transcribe",
                    files={"file": ("test.wav", audio_file, "audio/wav")},
                    data={"model": "nano", "language": "中文"},
                    timeout=30
                )
            
            if resp.status_code == 200:
                hypothesis = resp.json()["text"]
                references.append(reference)
                hypotheses.append(hypothesis)
        except:
            pass
        finally:
            os.unlink(temp_path)
    
    error_cer = cer(references, hypotheses)
    print(f"\nWenetSpeech CER: {error_cer*100:.2f}%")
    return error_cer


def main():
    parser = argparse.ArgumentParser(description="Fun-ASR WER/CER 测试")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--dataset", choices=["aishell", "wenetspeech", "all"], default="aishell")
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
    
    if args.dataset in ["aishell", "all"]:
        test_aishell(args.api_url, args.num_samples)
    
    if args.dataset in ["wenetspeech", "all"]:
        test_wenetspeech(args.api_url, args.num_samples)


if __name__ == "__main__":
    main()
