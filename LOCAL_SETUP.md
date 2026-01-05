# 本地部署 Fun-ASR

## 快速开始

### 方式1: 使用Conda (推荐，一键安装ffmpeg)

```bash
# 1. 创建并激活conda环境
conda create -n funasr python=3.8 -y
conda activate funasr

# 2. 一键启动
chmod +x run_conda.sh
./run_conda.sh
```

详细说明：[CONDA_SETUP.md](./CONDA_SETUP.md)

### 方式2: 使用venv (需要手动安装ffmpeg)

```bash
# 确保不在conda环境中
conda deactivate  # 如果在conda环境中

# 需要先安装ffmpeg（需要root权限）
# Ubuntu/Debian: sudo apt-get install ffmpeg
# CentOS/RHEL: sudo yum install ffmpeg

# 一键启动（自动创建环境、安装依赖、启动服务）
./run_local.sh
```

服务启动后访问：http://localhost:8000

## 测试

```bash
# 健康检查
curl http://localhost:8000/health

# 语音识别
curl -X POST http://localhost:8000/transcribe_batch \
  -F "files=@test.wav" \
  -F "model=mlt"
```

## 系统要求

- Python 3.8+
- 8GB+ RAM
- 推荐GPU（可选）
