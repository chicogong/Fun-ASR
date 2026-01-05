# Fun-ASR MLT 部署指南

快速在新机器上部署Fun-ASR MLT多语言语音识别服务。

## 快速开始

### 1. 获取代码

```bash
git clone https://github.com/chicogong/Fun-ASR.git
cd Fun-ASR
git checkout feature/mlt-batch-simple
```

### 2. Docker部署（推荐）

#### CPU模式（稳定推荐）

```bash
./docker-start.sh
```

#### GPU模式（需要兼容环境）

```bash
USE_GPU=true ./docker-start.sh
```

### 3. 本地部署

```bash
./run.sh local
```

## 详细说明

### Docker部署

**优点：**
- 环境隔离，不影响系统
- 一键启动，自动处理依赖
- 跨平台支持

**CPU模式（默认）：**
```bash
# 方式1：使用脚本（推荐）
./docker-start.sh

# 方式2：手动启动
docker build -t funasr-mlt-batch:latest .
docker run -d --name funasr -p 8000:8000 \
  -e USE_GPU=false \
  -v ~/.cache/modelscope:/root/.cache/modelscope \
  funasr-mlt-batch:latest
```

**GPU模式：**
```bash
# 需要：NVIDIA GPU + nvidia-docker2
USE_GPU=true ./docker-start.sh
```

**注意：** Docker GPU模式可能存在兼容性问题（PyTorch/CUDA版本），建议优先使用CPU模式或本地部署。

### 本地部署

**推荐环境：**
- Python 3.11+ （最佳兼容性）
- CUDA 12.1+ （GPU模式）

**一键启动：**
```bash
./run.sh local
```

脚本会自动：
1. 检测Python版本（优先3.11）
2. 创建虚拟环境venv311
3. 安装依赖
4. 启动服务

**手动安装：**
```bash
# 创建虚拟环境
python3.11 -m venv venv311
source venv311/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 启动服务
python server.py
```

## 验证部署

```bash
# 健康检查
curl http://localhost:8000/health

# 服务信息
curl http://localhost:8000/info

# 测试识别
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=zh"
```

## 环境要求

### 最低要求
- CPU: 2核+
- 内存: 8GB+
- 磁盘: 5GB+ (模型约2GB)

### GPU加速（可选）
- NVIDIA GPU (推荐Tesla T4或更好)
- CUDA 12.1+
- 显存: 4GB+

## 支持的语言

MLT模型支持31种语言，包括：

- 中文 (zh)
- 英文 (en)
- 日文 (ja)
- 韩文 (ko)
- 粤语 (yue)
- 越南语 (vi)
- 印尼语 (id)
- 泰语 (th)
- 马来语 (ms)
- 菲律宾语 (tl)
- 阿拉伯语 (ar)
- 印地语 (hi)
- 等...

## API使用

### 单文件识别

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=zh" \
  -F "hotwords=语音,识别"
```

### 批量识别

```bash
curl -X POST http://localhost:8000/transcribe_batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav" \
  -F "language=zh"
```

### API文档

访问: http://localhost:8000/docs

## 故障排除

### Docker相关

**容器启动失败：**
```bash
# 查看日志
docker logs funasr

# 重启容器
docker restart funasr
```

**GPU模式crash：**
```bash
# 使用CPU模式
docker stop funasr && docker rm funasr
./docker-start.sh  # 默认CPU模式
```

### 本地部署

**Python版本问题：**
```bash
# 确认Python 3.11已安装
python3.11 --version

# 如未安装，使用系统包管理器安装
# Ubuntu/Debian:
sudo apt install python3.11 python3.11-venv

# CentOS/RHEL:
sudo yum install python3.11
```

**FunASR模型加载失败：**
```bash
# 确保Python 3.11 + FunASR 1.2.9
source venv311/bin/activate
pip install "funasr==1.2.9" --force-reinstall
```

**GPU不可用：**
```bash
# 检查CUDA
nvidia-smi

# 检查PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 如GPU不可用，使用CPU模式
export USE_GPU=false
python server.py
```

## 性能优化

### 批处理优化
- 使用`/transcribe_batch`接口处理多文件
- 最大batch_size: 20

### GPU加速
- 本地部署推荐使用GPU（自动启用）
- RTF约0.44（实时因子，越小越快）

### 模型缓存
- 首次启动会下载约2GB模型
- 模型缓存在`~/.cache/modelscope/`
- Docker挂载此目录避免重复下载

## 生产部署建议

1. **使用本地部署** - GPU兼容性更好
2. **使用Nginx反向代理** - 负载均衡和SSL
3. **使用systemd管理** - 自动重启和日志
4. **监控GPU使用率** - nvidia-smi
5. **设置请求限流** - 防止过载

## 更多信息

- GitHub: https://github.com/chicogong/Fun-ASR
- 文档: README.md
- 问题反馈: GitHub Issues
