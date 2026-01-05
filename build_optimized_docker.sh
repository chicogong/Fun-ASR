#!/bin/bash
# 构建Fun-ASR Batch优化版Docker镜像

echo "🐳 构建Fun-ASR Batch优化版Docker镜像"
echo "========================================"

# 检查必需文件
required_files=("Dockerfile.optimized" "server_optimized.py" "model_batch.py" "model_mlt_batch.py")

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 缺少必需文件: $file"
        exit 1
    fi
done

echo "✅ 所有必需文件已准备"

# 构建镜像
echo ""
echo "🔨 开始构建Docker镜像..."
docker build -t funasr-batch-optimized:latest -f Dockerfile.optimized .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker镜像构建成功！"
    echo ""
    echo "📋 使用说明:"
    echo "  启动容器:"
    echo "    docker run -d -p 8000:8000 --name funasr-opt funasr-batch-optimized:latest"
    echo ""
    echo "  查看日志:"
    echo "    docker logs -f funasr-opt"
    echo ""
    echo "  测试API:"
    echo "    curl http://localhost:8000/health"
    echo ""
else
    echo "❌ Docker镜像构建失败"
    exit 1
fi
