#!/bin/bash
# Fun-ASR 本地一键启动脚本

set -e

echo "🚀 Fun-ASR 本地服务"
echo "===================="

# 检查是否在conda环境中
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  检测到conda环境: $CONDA_DEFAULT_ENV"
    echo "请先退出conda环境再运行此脚本："
    echo "  conda deactivate"
    echo ""
    echo "或者直接在conda环境中安装："
    echo "  conda install ffmpeg"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# 检查ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ 未安装 ffmpeg"
    echo "请先安装 ffmpeg："
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  CentOS/RHEL: sudo yum install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    exit 1
fi
echo "✅ ffmpeg: $(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f3)"

# 检测Python版本
PYTHON_CMD="python3"
if command -v python3.8 &> /dev/null; then
    PYTHON_CMD="python3.8"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
fi
echo "✅ Python: $($PYTHON_CMD --version)"

# 1. 检查并创建虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    $PYTHON_CMD -m venv venv
fi

# 2. 激活环境
echo "🔄 激活虚拟环境..."
source venv/bin/activate

# 验证激活成功
if [ "$VIRTUAL_ENV" == "" ]; then
    echo "❌ 虚拟环境激活失败"
    exit 1
fi
echo "✅ 虚拟环境: $VIRTUAL_ENV"

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
