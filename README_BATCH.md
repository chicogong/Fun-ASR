# Fun-ASR 批量推理服务器

基于 FunAudioLLM/Fun-ASR-MLT-Nano-2512 的批量优化推理服务器，支持 31 种语言。

## 快速开始

### 一键启动

```bash
./run.sh
```

脚本会自动：
- 创建 Python 虚拟环境
- 安装依赖
- 启动服务器（默认端口 8088）

### Docker 部署

```bash
docker build -t funasr-batch .
docker run -p 8088:8088 funasr-batch
```

## API 使用

### 单文件转写

```bash
curl -X POST http://localhost:8088/transcribe \
  -F "file=@audio.wav" \
  -F "language=zh"
```

### 批量转写（推荐）

```bash
curl -X POST http://localhost:8088/transcribe_batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav"
```

### 健康检查

```bash
curl http://localhost:8088/health
```

## 性能特点

- **批量优化**: batch_size=6 时达到 3.5x 加速
- **实时率**: RTF ~0.03 (33x 实时速度)
- **多语言**: 支持 31 种语言（中文、英语、日语、韩语等）
- **自动降级**: CUDA 不可用时自动切换到 CPU

## 环境变量

```bash
MODEL_PATH=FunAudioLLM/Fun-ASR-MLT-Nano-2512  # 模型路径
DEVICE=cuda:0                                  # 设备
BATCH_SIZE=6                                   # 最优批量大小
MAX_BATCH_SIZE=10                              # 最大批量限制
```

## 核心文件

- `server_batch.py` - FastAPI 批量推理服务器
- `model_batch.py` - 优化的 FunASRNano 模型
- `requirements-batch-server.txt` - 依赖列表
- `run.sh` - 一键启动脚本
- `Dockerfile` - Docker 镜像配置
