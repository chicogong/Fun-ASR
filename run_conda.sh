#!/bin/bash
# Fun-ASR Conda环境一键启动脚本

set -e

echo "🚀 Fun-ASR 本地服务 (Conda版)"
echo "================================"

# 检查是否在conda环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ 未检测到conda环境"
    echo "请先激活conda环境或创建新环境："
    echo "  conda create -n funasr python=3.8 -y"
    echo "  conda activate funasr"
    echo ""
    echo "然后重新运行此脚本"
    exit 1
fi

echo "✅ Conda环境: $CONDA_DEFAULT_ENV"
echo "✅ Python: $(python --version)"

# 检查并安装ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "📦 安装 ffmpeg..."
    conda install -c conda-forge ffmpeg -y
else
    echo "✅ ffmpeg: $(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f3)"
fi

# 安装/更新依赖
echo "📥 安装Python依赖..."
pip install -q --upgrade pip

# 先安装兼容版本的pydantic和fastapi
echo "  → 安装 pydantic 和 fastapi (兼容版本)..."
pip install -q 'pydantic>=1.10.0,<2.0.0' 'fastapi>=0.95.0,<0.100.0'

# 安装其他依赖
echo "  → 安装其他依赖..."
pip install -q -r requirements.txt 2>&1 | grep -v "WARNING: Ignoring invalid distribution" || true

# 检测设备
echo "🔍 检测设备..."
python -c "import torch; print('✅ CUDA 可用' if torch.cuda.is_available() else '⚠️  仅CPU模式')" 2>/dev/null || echo "⚠️  torch未安装或安装中"

# 启动服务
echo ""
echo "📡 启动服务..."
echo "================================"
uvicorn server_optimized:app --host 0.0.0.0 --port 8000
