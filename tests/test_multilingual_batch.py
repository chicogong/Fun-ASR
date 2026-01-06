"""
测试多语言Batch性能
"""
import sys
sys.path.insert(0, '..')

from model_batch import FunASRNano
import time
import os
from pathlib import Path
import glob

def test_language_performance(model, kwargs, lang_dir, lang_name):
    """测试单个语言的性能"""
    wav_files = sorted(glob.glob(os.path.join(lang_dir, "*.wav")))
    
    if len(wav_files) == 0:
        return None
    
    # 限制batch size
    batch_size = min(len(wav_files), 6)
    test_files = wav_files[:batch_size]
    
    # 计算总音频时长
    total_duration = 0
    for f in test_files:
        try:
            import wave
            with wave.open(f, 'r') as w:
                total_duration += w.getnframes() / float(w.getframerate())
        except:
            total_duration += 5  # 默认5秒
    
    # Batch推理
    start = time.time()
    results = model.inference(data_in=test_files, batch_size=batch_size, **kwargs)
    batch_time = time.time() - start
    
    return {
        'language': lang_name,
        'num_files': batch_size,
        'total_duration': total_duration,
        'batch_time': batch_time,
        'time_per_file': batch_time / batch_size,
        'rtf': batch_time / total_duration if total_duration > 0 else 0,
        'results': results[0] if results else []
    }

def main():
    print("=" * 70)
    print("🌐 多语言Batch性能测试")
    print("=" * 70)
    print()
    
    # 加载模型
    model_dir = os.path.expanduser("~/.cache/modelscope/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512")
    print("🔄 Loading MLT model...")
    
    from model_batch import FunASRNano
    model, kwargs = FunASRNano.from_pretrained(
        model=model_dir,
        device="cuda:0",
        disable_update=True
    )
    model.eval()
    print("✅ Model loaded!\n")
    
    # 查找测试数据
    test_data_dir = Path("test_data/multilingual")
    
    if not test_data_dir.exists():
        print("❌ 测试数据不存在！")
        print("   请先运行: python tests/download_multilingual_test_data.py")
        return
    
    lang_dirs = sorted([d for d in test_data_dir.iterdir() if d.is_dir()])
    
    if len(lang_dirs) == 0:
        print("❌ 没有找到测试数据")
        return
    
    print(f"📁 找到 {len(lang_dirs)} 个语言的测试数据\n")
    
    # 测试每种语言
    results = []
    print("=" * 70)
    print(f"{'语言':<10} {'文件数':<8} {'时长(s)':<10} {'耗时(s)':<10} {'RTF':<10}")
    print("=" * 70)
    
    for lang_dir in lang_dirs:
        lang_name = lang_dir.name
        result = test_language_performance(model, kwargs, str(lang_dir), lang_name)
        
        if result:
            results.append(result)
            print(f"{result['language']:<10} {result['num_files']:<8} "
                  f"{result['total_duration']:<10.1f} {result['batch_time']:<10.2f} "
                  f"{result['rtf']:<10.4f}")
            
            # 显示识别结果示例
            if result['results']:
                sample = result['results'][0].get('text', '')[:60]
                print(f"   示例: {sample}...")
            print()
    
    print("=" * 70)
    
    # 汇总统计
    if results:
        total_files = sum(r['num_files'] for r in results)
        total_duration = sum(r['total_duration'] for r in results)
        total_time = sum(r['batch_time'] for r in results)
        avg_rtf = total_time / total_duration if total_duration > 0 else 0
        
        print(f"\n📊 汇总:")
        print(f"   测试语言数: {len(results)}")
        print(f"   总文件数: {total_files}")
        print(f"   总音频时长: {total_duration:.1f}秒")
        print(f"   总处理时间: {total_time:.2f}秒")
        print(f"   平均RTF: {avg_rtf:.4f}")
        print(f"   处理速度: {1/avg_rtf:.1f}x 实时")
        
        # 计算每天处理能力
        if avg_rtf > 0:
            files_per_day = 86400 / (total_time / total_files)
            hours_per_day = (files_per_day * (total_duration / total_files)) / 3600
            print(f"\n💡 每天处理能力 (基于多语言测试):")
            print(f"   {hours_per_day:.0f} 小时/天")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
