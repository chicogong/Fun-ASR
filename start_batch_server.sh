#!/bin/bash
# Fun-ASR MLT Batch Server 一键启动脚本

set -e

MODE=${1:-local}
USE_VENV=${2:-auto}

echo "🚀 Fun-ASR MLT Batch Server"
echo "============================="
echo "📍 模式: $MODE"
echo ""

case "$MODE" in
  local)
    echo "🔧 本地模式启动"
    echo ""
    
    # 检测Python
    PYTHON_CMD="python"
    for cmd in python3.11 python3.10 python3.9 python3; do
      if command -v $cmd &> /dev/null; then
        PYTHON_CMD=$cmd
        break
      fi
    done
    echo "✅ Python: $($PYTHON_CMD --version)"
    
    # 激活虚拟环境
    if [ -d "venv311" ]; then
      source venv311/bin/activate
      echo "✅ 虚拟环境: venv311"
    elif [ -d "venv" ]; then
      source venv/bin/activate
      echo "✅ 虚拟环境: venv"
    fi
    
    echo ""
    echo "📦 安装依赖..."
    pip install -q fastapi uvicorn python-multipart 2>&1 | grep -v WARNING || true
    
    echo ""
    echo "📡 启动Batch Server..."
    echo "============================="
    echo "✅ 服务地址: http://localhost:8000"
    echo "📖 API文档: http://localhost:8000/docs"
    echo "📊 性能统计: http://localhost:8000/stats"
    echo ""
    
    # 启动服务
    $PYTHON_CMD server_batch.py
    ;;
    
  docker)
    echo "🐳 Docker模式启动"
    
    if ! command -v docker &> /dev/null; then
      echo "❌ Docker未安装"
      exit 1
    fi
    
    echo "📦 构建镜像..."
    docker build -t funasr-mlt-batch:latest -f Dockerfile.batch .
    
    echo "🚀 启动容器..."
    docker run -d \
      --name funasr-batch \
      --gpus all \
      -p 8000:8000 \
      -v ~/.cache/modelscope:/root/.cache/modelscope \
      funasr-mlt-batch:latest
    
    echo ""
    echo "✅ 容器已启动"
    echo "📊 查看日志: docker logs -f funasr-batch"
    echo "🛑 停止服务: docker stop funasr-batch && docker rm funasr-batch"
    ;;
    
  *)
    echo "用法: $0 {local|docker}"
    echo ""
    echo "示例:"
    echo "  $0 local    # 本地运行"
    echo "  $0 docker   # Docker运行"
    exit 1
    ;;
esac
