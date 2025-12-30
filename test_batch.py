"""测试 batch 推理"""
import time
import torch
from funasr import AutoModel

MODEL_DIR = "/data/extra/models/models--FunAudioLLM--Fun-ASR-Nano-2512/snapshots/a7088d620f755dcdca575b63db184c3ad55b2865"

def main():
    print("Loading model...")
    # 使用 model_batch.py 作为 remote_code
    model = AutoModel(
        model=MODEL_DIR,
        trust_remote_code=True,
        remote_code="./model_batch.py",
        device="cuda:0",
        disable_update=True,
    )

    # 测试音频
    audio_files = [
        f"{MODEL_DIR}/example/zh.mp3",
        f"{MODEL_DIR}/example/en.mp3",
        f"{MODEL_DIR}/example/ja.mp3",
    ]
    
    print(f"\n=== 测试 {len(audio_files)} 个文件 ===")
    
    # 单个推理 (对比基准)
    print("\n--- 串行推理 ---")
    start = time.time()
    for f in audio_files:
        res = model.generate(input=[f], cache={}, batch_size=1, language="auto")
        print(f"{f.split('/')[-1]}: {res[0]['text'][:50]}...")
    serial_time = time.time() - start
    print(f"串行耗时: {serial_time:.2f}s")
    
    # Batch 推理
    print("\n--- Batch 推理 ---")
    start = time.time()
    res = model.generate(input=audio_files, cache={}, batch_size=len(audio_files), language="auto")
    batch_time = time.time() - start
    
    for i, r in enumerate(res):
        print(f"{audio_files[i].split('/')[-1]}: {r['text'][:50]}...")
    print(f"Batch 耗时: {batch_time:.2f}s")
    
    print(f"\n=== 结果 ===")
    print(f"串行: {serial_time:.2f}s")
    print(f"Batch: {batch_time:.2f}s")
    print(f"加速比: {serial_time/batch_time:.2f}x")


if __name__ == "__main__":
    main()
