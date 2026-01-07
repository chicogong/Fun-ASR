#!/bin/bash
# RTF 快速测试脚本

set -e

AUDIO_FILE=${1:-test_02.wav}
SERVER=${2:-http://localhost:8088}

echo "🚀 Fun-ASR RTF 快速测试"
echo "========================"
echo ""

# 检查音频文件
if [ ! -f "$AUDIO_FILE" ]; then
    echo "❌ 音频文件不存在: $AUDIO_FILE"
    echo ""
    echo "用法: $0 <音频文件> [服务器地址]"
    echo "示例: $0 test_02.wav http://localhost:8088"
    exit 1
fi

# 检查服务器
echo "🔍 检查服务器状态..."
if curl -s "$SERVER/health" > /dev/null 2>&1; then
    echo "✅ 服务器运行正常"
else
    echo "❌ 服务器未运行或无法访问: $SERVER"
    echo "请先启动服务器: ./run.sh"
    exit 1
fi

echo ""
echo "📊 测试配置:"
echo "   音频文件: $AUDIO_FILE"
echo "   服务器: $SERVER"
echo ""

# 运行Python测试脚本
python test_rtf.py \
    --server "$SERVER" \
    --audio "$AUDIO_FILE" \
    --batch-sizes "1,3,6,10"
