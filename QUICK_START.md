# Fun-ASR MLT 快速开始

## 在任意机器上一键部署

### 方法1：直接使用Docker镜像（最快）

```bash
# 创建启动脚本
curl -O https://raw.githubusercontent.com/chicogong/Fun-ASR/feature/mlt-batch-simple/docker-start.sh
chmod +x docker-start.sh

# 一键启动（自动拉取CCR镜像）
./docker-start.sh
```

### 方法2：手动拉取运行

```bash
# CPU模式（推荐）
docker pull ccr.ccs.tencentyun.com/chico/funasr-mlt-batch:latest

docker run -d --name funasr -p 8000:8000 \
  -e USE_GPU=false \
  -v ~/.cache/modelscope:/root/.cache/modelscope \
  ccr.ccs.tencentyun.com/chico/funasr-mlt-batch:latest

# 等待服务启动（首次需要下载2GB模型）
docker logs -f funasr
```

### 方法3：从源码部署

```bash
# 克隆代码
git clone https://github.com/chicogong/Fun-ASR.git
cd Fun-ASR
git checkout feature/mlt-batch-simple

# Docker部署
./docker-start.sh

# 或本地部署
./run.sh local
```

## 验证服务

```bash
# 健康检查
curl http://localhost:8000/health
# 返回: {"status":"ok","model":"MLT","batch_optimized":true}

# 服务信息
curl http://localhost:8000/info

# API文档
浏览器访问: http://localhost:8000/docs
```

## 使用示例

### 单文件识别

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=zh"
```

### 批量识别

```bash
curl -X POST http://localhost:8000/transcribe_batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav" \
  -F "language=zh"
```

### Python调用示例

```python
import requests

# 单文件识别
url = "http://localhost:8000/transcribe"
files = {"file": open("audio.wav", "rb")}
data = {"language": "zh"}
response = requests.post(url, files=files, data=data)
print(response.json())

# 批量识别
url = "http://localhost:8000/transcribe_batch"
files = [
    ("files", open("audio1.wav", "rb")),
    ("files", open("audio2.wav", "rb")),
]
data = {"language": "zh"}
response = requests.post(url, files=files, data=data)
print(response.json())
```

## 支持的语言

31种语言：zh, en, ja, ko, yue, vi, id, th, ms, tl, ar, hi, bg, hr, cs, da, nl, et, fi, el, hu, ga, lv, lt, mt, pl, pt, ro, sk, sl, sv

## 环境变量

- `USE_GPU=true` - 使用GPU模式（**默认，推荐，性能提升5倍**）
- `USE_GPU=false` - 使用CPU模式
- `MODEL_PATH` - 模型路径（默认：FunAudioLLM/Fun-ASR-MLT-Nano-2512）
- `MAX_BATCH_SIZE` - 最大批处理大小（默认：20）

**性能对比**: GPU 模式 RTF ≈ 0.06，CPU 模式 RTF ≈ 0.31（GPU 快 5 倍）

## 常用命令

```bash
# 查看日志
docker logs -f funasr

# 重启服务
docker restart funasr

# 停止服务
docker stop funasr && docker rm funasr

# 查看GPU使用（如果使用GPU）
nvidia-smi
```

## 镜像信息

- **最新版本**: ccr.ccs.tencentyun.com/chico/funasr-mlt-batch:latest
- **Registry**: 腾讯云CCR
- **大小**: ~8.5GB
- **基础镜像**: ccr.ccs.tencentyun.com/chico/funasr-server:latest

## 故障排除

### 容器启动失败

```bash
# 查看详细日志
docker logs funasr

# 检查端口占用
lsof -i:8000

# 使用其他端口
docker run -d --name funasr -p 8001:8000 ...
```

### 模型下载慢

```bash
# 模型会缓存在 ~/.cache/modelscope/
# 首次下载约2GB，后续启动会使用缓存
ls -lh ~/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512/
```

### GPU模式问题

```bash
# 如果GPU模式crash，使用CPU模式
docker run -d --name funasr -p 8000:8000 \
  -e USE_GPU=false \
  ccr.ccs.tencentyun.com/chico/funasr-mlt-batch:latest
```

## 更多信息

- **完整文档**: [DEPLOY.md](DEPLOY.md)
- **GitHub**: https://github.com/chicogong/Fun-ASR
- **问题反馈**: GitHub Issues
