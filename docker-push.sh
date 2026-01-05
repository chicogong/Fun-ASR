#!/bin/bash
# 推送镜像到腾讯云CCR容器镜像服务

set -e

echo "🚀 推送Fun-ASR MLT镜像到CCR"
echo "============================="

# CCR配置
REGISTRY="ccr.ccs.tencentyun.com"
NAMESPACE="chico"
USERNAME="100014078614"
PASSWORD="lty520.."

# 镜像信息
LOCAL_IMAGE="funasr-mlt-batch:latest"
REMOTE_IMAGE="${REGISTRY}/${NAMESPACE}/funasr-mlt-batch:latest"
REMOTE_IMAGE_TAG="${REGISTRY}/${NAMESPACE}/funasr-mlt-batch:$(date +%Y%m%d-%H%M%S)"

echo "📋 配置信息："
echo "   本地镜像: $LOCAL_IMAGE"
echo "   远程镜像: $REMOTE_IMAGE"
echo "   带时间戳: $REMOTE_IMAGE_TAG"
echo ""

# 检查本地镜像是否存在
if ! docker images | grep -q "^funasr-mlt-batch"; then
    echo "📦 本地镜像不存在，开始构建..."
    docker build -t $LOCAL_IMAGE .
    echo "✅ 镜像构建完成"
else
    echo "✅ 本地镜像已存在"
fi

# 登录CCR
echo ""
echo "🔐 登录CCR..."
echo "$PASSWORD" | docker login $REGISTRY -u $USERNAME --password-stdin

if [ $? -eq 0 ]; then
    echo "✅ 登录成功"
else
    echo "❌ 登录失败，请检查用户名和密码"
    exit 1
fi

# 打标签
echo ""
echo "🏷️  打标签..."
docker tag $LOCAL_IMAGE $REMOTE_IMAGE
docker tag $LOCAL_IMAGE $REMOTE_IMAGE_TAG

echo "✅ 标签已创建："
echo "   $REMOTE_IMAGE"
echo "   $REMOTE_IMAGE_TAG"

# 推送镜像
echo ""
echo "📤 推送镜像到CCR..."
docker push $REMOTE_IMAGE
docker push $REMOTE_IMAGE_TAG

echo ""
echo "✅ 推送完成！"
echo ""
echo "🌐 镜像地址："
echo "   最新版本: $REMOTE_IMAGE"
echo "   带时间戳: $REMOTE_IMAGE_TAG"
echo ""
echo "📋 在其他机器上使用："
echo "   docker pull $REMOTE_IMAGE"
echo "   docker run -d --name funasr -p 8000:8000 \\"
echo "     -e USE_GPU=false \\"
echo "     -v ~/.cache/modelscope:/root/.cache/modelscope \\"
echo "     $REMOTE_IMAGE"
echo ""
