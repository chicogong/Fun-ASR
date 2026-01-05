#!/bin/bash
# Fun-ASR 本地一键启动脚本

set -e

echo "🚀 Fun-ASR 本地服务"
echo "===================="

# 1. 检查并创建虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
fi

# 2. 激活环境
source venv/bin/activate

# 3. 安装/更新依赖
echo "📥 安装依赖..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 4. 检测设备
echo "🔍 检测设备..."
python3 -c "import torch; print('✅ CUDA' if torch.cuda.is_available() else '⚠️  CPU')"

# 5. 启动服务
echo "📡 启动服务..."
echo "===================="
uvicorn server_optimized:app --host 0.0.0.0 --port 8000
