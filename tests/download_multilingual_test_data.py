"""
下载多语言测试数据 - 使用FLEURS数据集
"""
import os
import soundfile as sf
import numpy as np
from datasets import load_dataset
from pathlib import Path

def download_samples():
    """��载多语言测试样本"""
    print("=" * 70)
    print("🌐 下载FLEURS多语言测试数据")
    print("=" * 70)
    print()
    
    output_dir = Path("test_data/multilingual")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试语言子集（FLEURS支持的语言）
    test_languages = [
        ("zh_cn", "中文", 5),
        ("en_us", "英语", 5),
        ("ja_jp", "日语", 5),
        ("ko_kr", "韩语", 5),
        ("fr_fr", "法语", 5),
        ("de_de", "德语", 5),
        ("es_419", "西班牙语", 5),
        ("ru_ru", "俄语", 5),
        ("ar_ar", "阿拉伯语", 5),
        ("hi_in", "印地语", 5),
    ]
    
    all_downloaded = []
    
    for lang_code, lang_name, num_samples in test_languages:
        print(f"📥 下载 {lang_name} ({lang_code})...")
        
        try:
            # 加载FLEURS数据集
            dataset = load_dataset(
                "google/fleurs",
                lang_code,
                split=f"test[:{num_samples}]",
                trust_remote_code=True
            )
            
            # 保存音频文件
            lang_dir = output_dir / lang_name
            lang_dir.mkdir(exist_ok=True)
            
            for i, item in enumerate(dataset):
                audio = item['audio']
                text = item.get('transcription', '')
                
                # 保存音频
                audio_path = lang_dir / f"{lang_code}_{i}.wav"
                sf.write(audio_path, audio['array'], audio['sampling_rate'])
                
                # 保存对应的文本（用于验证准确性）
                text_path = lang_dir / f"{lang_code}_{i}.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"   ✅ {audio_path.name} ({text[:50]}...)")
                
                all_downloaded.append(str(audio_path))
            
            print(f"   ✅ 完成: {num_samples} 个样本\n")
            
        except Exception as e:
            print(f"   ❌ 错误: {e}\n")
            continue
    
    print("=" * 70)
    print(f"✅ 总共下载: {len(all_downloaded)} 个音频文件")
    print(f"📁 保存位置: {output_dir}")
    print("=" * 70)
    
    return all_downloaded

if __name__ == "__main__":
    download_samples()
