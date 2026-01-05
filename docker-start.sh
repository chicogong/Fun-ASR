#!/bin/bash
# Fun-ASR MLT Docker 启动脚本
# 用于在其他机器上快速部署

set -e

echo "🚀 Fun-ASR MLT Batch Server - Docker部署"
echo "=========================================="

# 配置
IMAGE_NAME="funasr-mlt-batch"
CONTAINER_NAME="funasr"
PORT=8000
USE_GPU=${USE_GPU:-false}  # 默认CPU模式（GPU模式需要兼容的环境）

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

echo "✅ Docker已安装"

# 停止并删除旧容器
if docker ps -a | grep -q $CONTAINER_NAME; then
    echo "🧹 清理旧容器..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi

# 检查镜像是否存在
if ! docker images | grep -q "^$IMAGE_NAME"; then
    echo "📦 镜像不存在，开始构建..."
    if [ ! -f "Dockerfile" ]; then
        echo "❌ 找不到Dockerfile，请确保在项目根目录运行此脚本"
        exit 1
    fi
    docker build -t $IMAGE_NAME:latest .
    echo "✅ 镜像构建完成"
else
    echo "✅ 镜像已存在: $IMAGE_NAME"
fi

# 构建docker run命令
DOCKER_CMD="docker run -d --name $CONTAINER_NAME"

# GPU选项
if [ "$USE_GPU" = "true" ]; then
    echo "🎮 启用GPU模式（可能需要兼容的CUDA环境）"
    DOCKER_CMD="$DOCKER_CMD --gpus all"
    GPU_ENV="-e USE_GPU=true"
else
    echo "💻 使用CPU模式（稳定，推荐）"
    GPU_ENV="-e USE_GPU=false"
fi

# 端口映射
DOCKER_CMD="$DOCKER_CMD -p $PORT:8000"

# 环境变量
DOCKER_CMD="$DOCKER_CMD $GPU_ENV"
DOCKER_CMD="$DOCKER_CMD -e MODEL_PATH=FunAudioLLM/Fun-ASR-MLT-Nano-2512"
DOCKER_CMD="$DOCKER_CMD -e MAX_BATCH_SIZE=20"

# 挂载模型缓存（加速启动）
DOCKER_CMD="$DOCKER_CMD -v ~/.cache/modelscope:/root/.cache/modelscope"

# 镜像名
DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME:latest"

echo ""
echo "📋 启动命令："
echo "$DOCKER_CMD"
echo ""

# 启动容器
eval $DOCKER_CMD

echo ""
echo "⏳ 等待容器启动..."
sleep 5

# 检查容器状态
if docker ps | grep -q $CONTAINER_NAME; then
    echo "✅ 容器启动成功！"
    echo ""
    echo "📊 容器状态："
    docker ps | grep $CONTAINER_NAME
    echo ""
    echo "⏳ 等待模型加载（首次启动需要下载约2GB模型）..."
    echo "   可以使用以下命令查看日志："
    echo "   docker logs -f $CONTAINER_NAME"
    echo ""

    # 等待服务就绪
    echo "🔍 检测服务状态..."
    for i in {1..60}; do
        if curl -s http://localhost:$PORT/health &>/dev/null; then
            echo ""
            echo "✅ 服务已就绪！"
            echo ""
            echo "🌐 服务地址："
            echo "   - API文档: http://localhost:$PORT/docs"
            echo "   - 健康检查: http://localhost:$PORT/health"
            echo "   - 服务信息: http://localhost:$PORT/info"
            echo ""
            echo "📝 使用示例："
            echo "   # 健康检查"
            echo "   curl http://localhost:$PORT/health"
            echo ""
            echo "   # 单文件识别"
            echo "   curl -X POST http://localhost:$PORT/transcribe \\"
            echo "     -F \"file=@audio.wav\" \\"
            echo "     -F \"language=zh\""
            echo ""
            echo "   # 批量识别"
            echo "   curl -X POST http://localhost:$PORT/transcribe_batch \\"
            echo "     -F \"files=@audio1.wav\" \\"
            echo "     -F \"files=@audio2.wav\" \\"
            echo "     -F \"language=zh\""
            echo ""
            echo "🛑 停止服务："
            echo "   docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"
            echo ""
            exit 0
        fi
        echo -n "."
        sleep 2
    done

    echo ""
    echo "⚠️  服务启动超时，请检查日志："
    echo "   docker logs $CONTAINER_NAME"

else
    echo "❌ 容器启动失败！"
    echo ""
    echo "📋 查看日志："
    docker logs $CONTAINER_NAME
    echo ""
    echo "💡 提示："
    echo "   - GPU模式可能需要兼容的CUDA环境"
    echo "   - 建议先尝试CPU模式: USE_GPU=false ./docker-start.sh"
    exit 1
fi
