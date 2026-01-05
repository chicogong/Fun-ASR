#!/usr/bin/env python3
"""使用 jiwer 进行完整的 AISHELL-1 WER/CER 测试"""

import os
import json
import requests
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from jiwer import wer, cer, process_words, process_characters

# 配置
API_URL = "http://localhost:8000/transcribe"
MANIFEST_PATH = "/data/workspace/Fun-ASR/tests/aishell_test/test_manifest_full.json"
RESULTS_PATH = "/data/workspace/Fun-ASR/tests/aishell_test/full_wer_results.json"
BATCH_SIZE = 20  # 并发数

def normalize_text(text):
    """规范化文本：去除空格和标点"""
    if not text:
        return ""
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[。，！？、；：""''《》【】（）\.\,\!\?\;\:\"\'\(\)\[\]·]', '', text)
    return text

def transcribe_file(item):
    """识别单个文件"""
    audio_path = item['audio']
    try:
        with open(audio_path, 'rb') as f:
            files = {'file': (os.path.basename(audio_path), f, 'audio/wav')}
            data = {'language': '中文', 'model': 'nano'}
            response = requests.post(API_URL, files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return {
                'id': item['id'],
                'reference': item['reference'],
                'hypothesis': result.get('text', ''),
                'success': True
            }
        else:
            return {
                'id': item['id'],
                'reference': item['reference'],
                'hypothesis': '',
                'success': False,
                'error': f"HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            'id': item['id'],
            'reference': item['reference'],
            'hypothesis': '',
            'success': False,
            'error': str(e)
        }

def main():
    # 加载测试数据
    print("Loading test manifest...")
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Total samples: {len(test_data)}")
    
    # 批量识别
    print(f"\nTranscribing with {BATCH_SIZE} concurrent workers...")
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = {executor.submit(transcribe_file, item): item for item in test_data}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            if completed % 500 == 0:
                elapsed = time.time() - start_time
                speed = completed / elapsed
                eta = (len(test_data) - completed) / speed
                print(f"  Progress: {completed}/{len(test_data)} ({100*completed/len(test_data):.1f}%), "
                      f"Speed: {speed:.1f} files/s, ETA: {eta:.0f}s")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({len(results)/elapsed:.1f} files/s)")
    
    # 统计成功率
    success_results = [r for r in results if r['success']]
    print(f"Success: {len(success_results)}/{len(results)} ({100*len(success_results)/len(results):.1f}%)")
    
    # 准备 jiwer 输入
    references = []
    hypotheses = []
    
    for r in success_results:
        ref = normalize_text(r['reference'])
        hyp = normalize_text(r['hypothesis'])
        if ref:  # 只计算有参考文本的
            references.append(ref)
            hypotheses.append(hyp)
    
    print(f"\nValid samples for WER calculation: {len(references)}")
    
    # 使用 jiwer 计算 CER
    print("\nCalculating CER with jiwer...")
    
    # 字符级别 (CER)
    cer_score = cer(references, hypotheses)
    
    # 详细的字符级统计
    char_output = process_characters(references, hypotheses)
    
    print("\n" + "="*60)
    print("           AISHELL-1 完整测试集 WER/CER 结果")
    print("="*60)
    print(f"\n测试集: AISHELL-1 Test Set")
    print(f"样本数: {len(references)}")
    print(f"模型: Fun-ASR-Nano")
    print(f"\n{'='*40}")
    print(f"  字符错误率 (CER): {cer_score*100:.2f}%")
    print(f"  准确率: {(1-cer_score)*100:.2f}%")
    print(f"{'='*40}")
    
    print(f"\n详细统计 (jiwer):")
    print(f"  Substitutions: {char_output.substitutions}")
    print(f"  Deletions: {char_output.deletions}")
    print(f"  Insertions: {char_output.insertions}")
    print(f"  Hits: {char_output.hits}")
    
    # 保存结果
    summary = {
        'dataset': 'AISHELL-1',
        'model': 'Fun-ASR-Nano',
        'num_samples': len(references),
        'cer': cer_score * 100,
        'accuracy': (1 - cer_score) * 100,
        'substitutions': char_output.substitutions,
        'deletions': char_output.deletions,
        'insertions': char_output.insertions,
        'hits': char_output.hits,
        'processing_time': elapsed,
        'throughput': len(results) / elapsed
    }
    
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {RESULTS_PATH}")
    
    return summary

if __name__ == "__main__":
    main()
