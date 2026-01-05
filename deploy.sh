#!/bin/bash
# Fun-ASR MLT 便携部署脚本 - 可在任何Linux机器上运行

set -e

echo "🚀 Fun-ASR MLT 多语言模型一键部署"
echo "支持31种语言的高精度语音识别"
echo ""

# 检查系统
if ! command -v docker &> /dev/null; then
    echo "❌ 未检测到Docker，正在安装..."

    # 检测系统类型并安装Docker
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case $ID in
            ubuntu|debian)
                sudo apt-get update
                sudo apt-get install -y docker.io
                sudo systemctl start docker
                sudo systemctl enable docker
                ;;
            centos|rhel|fedora)
                sudo yum install -y docker
                sudo systemctl start docker
                sudo systemctl enable docker
                ;;
            *)
                echo "请手动安装Docker后重新运行此脚本"
                exit 1
                ;;
        esac
    else
        echo "无法识别系统类型，请手动安装Docker"
        exit 1
    fi
fi

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "🔧 启动Docker服务..."
    sudo systemctl start docker 2>/dev/null || service docker start 2>/dev/null || echo "请手动启动Docker服务"
fi

echo "🧹 清理现有服务..."
# 停止可能存在的容器
docker stop funasr-server-dual funasr-server funasr-mlt funasr-nano 2>/dev/null || true
docker rm funasr-server-dual funasr-server funasr-mlt funasr-nano 2>/dev/null || true

# 强制清理端口
lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
netstat -tulpn 2>/dev/null | grep :8000 | awk '{print $7}' | cut -d'/' -f1 | xargs kill -9 2>/dev/null || true

echo "📥 拉取最新镜像..."
docker pull ccr.ccs.tencentyun.com/chico/funasr-server:latest

sleep 3

echo "🚀 启动Fun-ASR MLT服务..."
docker run -d \
  --name funasr-server-dual \
  --restart unless-stopped \
  -p 8000:8000 \
  -e LOAD_NANO=true \
  -e LOAD_MLT=true \
  ccr.ccs.tencentyun.com/chico/funasr-server:latest

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 服务启动成功！"
    echo ""
    echo "⏳ 首次启动需要下载模型（约5-10分钟），请耐心等待..."
    echo ""
    echo "🌐 服务地址:"
    echo "   API文档:    http://$(hostname -I | awk '{print $1}'):8000/docs"
    echo "   健康检查:   http://$(hostname -I | awk '{print $1}'):8000/health"
    echo "   模型列表:   http://$(hostname -I | awk '{print $1}'):8000/models"
    echo ""
    echo "🔗 本地访问:"
    echo "   http://localhost:8000/docs"
    echo ""
    echo "📋 管理命令:"
    echo "   查看日志:   docker logs -f funasr-server-dual"
    echo "   停止服务:   docker stop funasr-server-dual"
    echo "   重启服务:   docker restart funasr-server-dual"
    echo ""
    echo "🧪 测试命令:"
    echo "   curl http://localhost:8000/health"
    echo ""
    echo "📝 支持模型:"
    echo "   nano: 中文优化（中文、英文、日文、粤语、韩文）"
    echo "   mlt:  31种语言支持"
else
    echo "❌ 服务启动失败，请检查Docker状态"
    exit 1
fi