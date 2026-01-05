#!/bin/bash
# 测试Docker优化版本的batch性能

echo "🧪 Fun-ASR Docker优化版本 - Batch性能测试"
echo "=========================================="

# 检查容器是否运行
if ! docker ps | grep -q funasr-opt; then
    echo "❌ 容器未运行。请先启动："
    echo "   docker run -d -p 8000:8000 --name funasr-opt funasr-batch-optimized:latest"
    exit 1
fi

echo "✅ 容器正在运行"
echo ""

# 等待服务就绪
echo "⏳ 等待服务就绪..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ 服务已就绪"
        break
    fi
    sleep 2
done

echo ""
echo "📊 健康检查:"
curl -s http://localhost:8000/health | python3 -m json.tool
echo ""

# 创建测试音频文件
echo "📝 创建测试音频文件..."
python3 << 'EOPYTHON'
import numpy as np
import soundfile as sf
import tempfile
import os

def create_test_audio(duration=1.0):
    t = np.linspace(0, duration, int(16000 * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    return audio.astype(np.float32)

# 创建6个测试文件
test_files = []
for i in range(6):
    audio = create_test_audio(1.0)
    filename = f"/tmp/test_audio_{i}.wav"
    sf.write(filename, audio, 16000)
    test_files.append(filename)
    print(f"  创建: {filename}")

print(f"\n✅ 创建了 {len(test_files)} 个测试文件")
EOPYTHON

if [ $? -ne 0 ]; then
    echo "❌ 创建测试文件失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "🚀 开始Batch性能测试"
echo "=========================================="

# 测试函数
test_batch() {
    local model=$1
    local batch_size=$2

    echo ""
    echo "[$model Model - batch_size=$batch_size]"

    # 构建文件参数
    local files=""
    for i in $(seq 0 $((batch_size-1))); do
        files="$files -F files=@/tmp/test_audio_$i.wav"
    done

    # 测试
    local start=$(date +%s.%N)
    local response=$(curl -s -X POST http://localhost:8000/transcribe_batch \
        $files \
        -F "model=$model" 2>&1)
    local end=$(date +%s.%N)

    local elapsed=$(echo "$end - $start" | bc)
    local rtf=$(echo "scale=4; $elapsed / $batch_size" | bc)

    if echo "$response" | grep -q "results"; then
        local count=$(echo "$response" | grep -o "filename" | wc -l)
        echo "  ✅ 成功 | 时间:${elapsed}s | RTF:$rtf | 处理:$count 文件"
    else
        echo "  ❌ 失败"
        echo "$response" | head -3
    fi
}

# 测试Nano模型
echo ""
echo "=== Nano Model (中文优化) ==="
for bs in 1 2 4 6; do
    test_batch "nano" $bs
done

# 测试MLT模型
echo ""
echo "=== MLT Model (多语言) ==="
for bs in 1 2 3 4; do
    test_batch "mlt" $bs
done

echo ""
echo "=========================================="
echo "✅ 测试完成"
echo "=========================================="

# 清理测试文件
echo ""
echo "🧹 清理测试文件..."
rm -f /tmp/test_audio_*.wav
echo "✅ 清理完成"
