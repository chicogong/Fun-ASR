#!/bin/bash
# Fun-ASR 一键启动脚本（支持conda和venv）

set -e

echo "🚀 Fun-ASR 本地服务"
echo "===================="

# 检测Python
PYTHON_CMD="python"
if command -v python3.8 &> /dev/null; then
    PYTHON_CMD="python3.8"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
fi

echo "✅ Python: $($PYTHON_CMD --version)"

# 如果在conda环境中
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    echo "✅ Conda环境: $CONDA_DEFAULT_ENV"

    # 检查并安装ffmpeg
    if ! command -v ffmpeg &> /dev/null; then
        echo "📦 安装 ffmpeg..."
        conda install -c conda-forge ffmpeg -y
    fi
else
    # venv环境
    if [ ! -d "venv" ]; then
        echo "📦 创建虚拟环境..."
        $PYTHON_CMD -m venv venv
    fi
    source venv/bin/activate
    echo "✅ 虚拟环境: $VIRTUAL_ENV"
fi

# 安装依赖
echo "📥 安装依赖..."
pip install -q --upgrade pip
pip install -q 'pydantic>=1.10.0,<2.0.0' 'fastapi>=0.95.0,<0.100.0' 2>&1 | grep -v WARNING || true
pip install -q -r requirements.txt 2>&1 | grep -v WARNING || true

# 检测设备
echo "🔍 检测设备..."
python -c "import torch; print('✅ CUDA' if torch.cuda.is_available() else '⚠️  CPU')" 2>/dev/null || echo "⚠️  torch安装中..."

# 启动服务
echo "📡 启动服务..."
echo "===================="
uvicorn server_optimized:app --host 0.0.0.0 --port 8000
