#!/bin/bash
# Fun-ASR 本地一键启动脚本

set -e

# 检测Python版本
PYTHON_CMD="python3"
if command -v python3.8 &> /dev/null; then
    PYTHON_CMD="python3.8"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
fi

echo "🚀 Fun-ASR 本地服务"
echo "===================="
echo "Using: $($PYTHON_CMD --version)"

# 1. 检查并创建虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    $PYTHON_CMD -m venv venv
fi

# 2. 激活环境
source venv/bin/activate

# 3. 安装/更新依赖
echo "📥 安装依赖..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 4. 检测设备
echo "🔍 检测设备..."
python -c "import torch; print('✅ CUDA' if torch.cuda.is_available() else '⚠️  CPU')"

# 5. 启动服务
echo "📡 启动服务..."
echo "===================="
uvicorn server_optimized:app --host 0.0.0.0 --port 8000
