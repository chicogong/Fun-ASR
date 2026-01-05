#!/bin/bash
# Fun-ASR 统一启动脚本（本地/Docker）

set -e

MODE=${1:-local}

echo "🚀 Fun-ASR MLT Batch Server"
echo "============================="

case "$MODE" in
  local)
    echo "📍 模式: 本地运行"

    # 检测Python
    PYTHON_CMD="python"
    for cmd in python3.10 python3.9 python3.8 python3; do
      if command -v $cmd &> /dev/null; then
        PYTHON_CMD=$cmd
        break
      fi
    done

    echo "✅ Python: $($PYTHON_CMD --version)"

    # Conda环境
    if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
      echo "✅ Conda环境: $CONDA_DEFAULT_ENV"

      # 检查ffmpeg
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
    pip install -q 'pydantic>=1.10.0,<2.0.0' 'fastapi>=0.95.0' 2>&1 | grep -v WARNING || true
    pip install -q -r requirements.txt 2>&1 | grep -v WARNING || true

    # 检测设备
    echo "🔍 检测设备..."
    python -c "import torch; print('✅ CUDA' if torch.cuda.is_available() else '⚠️  CPU')" 2>/dev/null || echo "⚠️  torch安装中..."

    # 启动服务
    echo ""
    echo "📡 启动服务..."
    echo "============================="
    uvicorn server:app --host 0.0.0.0 --port 8000
    ;;

  docker)
    echo "📍 模式: Docker运行"

    if ! command -v docker &> /dev/null; then
      echo "❌ Docker未安装"
      exit 1
    fi

    echo "🐳 构建Docker镜像..."
    docker build -t funasr-mlt-batch:latest .

    echo "🚀 启动容器..."
    docker run -d \
      --name funasr \
      --gpus all \
      -p 8000:8000 \
      -v ~/.cache/modelscope:/root/.cache/modelscope \
      funasr-mlt-batch:latest

    echo ""
    echo "✅ 容器已启动: funasr"
    echo "📊 查看日志: docker logs -f funasr"
    echo "🛑 停止服务: docker stop funasr && docker rm funasr"
    ;;

  compose)
    echo "📍 模式: Docker Compose运行"

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
      echo "❌ Docker Compose未安装"
      exit 1
    fi

    echo "🐳 启动服务..."
    docker-compose up -d

    echo ""
    echo "✅ 服务已启动"
    echo "📊 查看日志: docker-compose logs -f"
    echo "🛑 停止服务: docker-compose down"
    ;;

  *)
    echo "用法: $0 {local|docker|compose}"
    echo ""
    echo "  local   - 本地运行（conda或venv）"
    echo "  docker  - Docker运行"
    echo "  compose - Docker Compose运行"
    exit 1
    ;;
esac

echo ""
echo "✅ 服务地址: http://localhost:8000"
echo "📖 API文档: http://localhost:8000/docs"
