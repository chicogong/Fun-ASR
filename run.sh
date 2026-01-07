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
    for cmd in python3.11 python3.10 python3.9 python3.8 python3; do
      if command -v $cmd &> /dev/null; then
        PYTHON_CMD=$cmd
        break
      fi
    done
    echo "✅ Python: $($PYTHON_CMD --version)"

    # 检查是否在虚拟环境中
    if [ -z "$VIRTUAL_ENV" ]; then
      echo "🔧 检测虚拟环境..."

      # 优先使用venv311
      if [ -d "venv311" ]; then
        source venv311/bin/activate
        echo "✅ 激活: venv311"
      elif [ -d "venv" ]; then
        source venv/bin/activate
        echo "✅ 激活: venv"
      elif [ -d "env-3.8.8" ]; then
        source env-3.8.8/bin/activate
        echo "✅ 激活: env-3.8.8"
      else
        # 创建新的虚拟环境
        echo "📦 创建虚拟环境: venv"
        $PYTHON_CMD -m venv venv
        source venv/bin/activate
        echo "✅ 虚拟环境已创建"
      fi
    else
      echo "✅ 已在虚拟环境中: $VIRTUAL_ENV"
    fi

    echo ""
    echo "📦 安装/更新依赖..."

    # 升级pip
    python -m pip install --upgrade pip -q 2>&1 | grep -v WARNING || true

    # 优先使用batch server的requirements
    if [ -f "requirements-batch-server.txt" ]; then
      echo "   安装 Batch Server 依赖..."
      pip install -q -r requirements-batch-server.txt 2>&1 | grep -v WARNING || true
    elif [ -f "requirements.txt" ]; then
      echo "   安装 requirements.txt..."
      pip install -q -r requirements.txt 2>&1 | grep -v WARNING || true
    fi

    # 确保FastAPI依赖已安装
    echo "   确保 FastAPI 依赖..."
    pip install -q fastapi uvicorn[standard] python-multipart 2>&1 | grep -v WARNING || true

    # 确保测试依赖已安装
    echo "   安装测试依赖..."
    pip install -q datasets soundfile 2>&1 | grep -v WARNING || true

    echo "   ✅ 依赖安装完成"
    echo ""

    # 验证关键依赖
    echo "🔍 验证依赖..."
    python -c "import fastapi; print('   ✅ FastAPI:', fastapi.__version__)" 2>&1 || echo "   ⚠️  FastAPI 未安装"
    python -c "import uvicorn; print('   ✅ Uvicorn:', uvicorn.__version__)" 2>&1 || echo "   ⚠️  Uvicorn 未安装"
    echo ""

    echo "📡 启动Batch Server..."
    echo "============================="
    echo "✅ 服务地址: http://localhost:8000"
    echo "📖 API文档: http://localhost:8000/docs"
    echo "📊 性能统计: http://localhost:8000/stats"
    echo ""

    # 启动服务
    python server_batch.py
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
