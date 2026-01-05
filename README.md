# Fun-ASR MLT Batch

Fun-ASR多语言模型（MLT）的batch批处理优化版本，支持31种语言的语音识别。

## 特性

- ✅ **MLT多语言模型**：支持31种语言
- ✅ **Batch批处理优化**：显著提升吞吐量
- ✅ **一键启动**：本地/Docker统一部署
- ✅ **简洁架构**：核心代码，易于维护

## 快速开始

### 本地运行

```bash
# 一键启动（自动安装依赖）
./run.sh local
```

### Docker运行

```bash
# 方式1: Docker
./run.sh docker

# 方式2: Docker Compose
./run.sh compose
```

## 使用示例

### 单文件识别

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=zh"
```

### 批量识别（batch优化）

```bash
curl -X POST http://localhost:8000/transcribe_batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav" \
  -F "language=zh"
```

### 健康检查

```bash
curl http://localhost:8000/health
```

## API文档

启动服务后访问：http://localhost:8000/docs

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_PATH` | 模型路径 | `FunAudioLLM/Fun-ASR-MLT-Nano-2512` |
| `MAX_BATCH_SIZE` | 最大batch大小 | `20` |
| `USE_GPU` | 是否使用GPU | `true` (默认GPU，设置false使用CPU) |

## 系统要求

- **Python 3.11+** (推荐，MLT模型最佳兼容性)
- Python 3.8+ (也支持，但可能需要额外配置)
- 8GB+ RAM
- **GPU推荐** (默认启用CUDA，支持CPU模式)

## 性能

- **RTF (Real-Time Factor)**：约0.44（batch_size=3）
- **吞吐量提升**：相比单文件处理提升约14%

## 文件结构

```
Fun-ASR/
├── server.py              # MLT batch服务器（使用AutoModel，无需自定义wrapper）
├── Dockerfile             # Docker镜像
├── docker-compose.yml     # Docker Compose配置
├── run.sh                 # 统一启动脚本
├── requirements.txt       # Python依赖
└── README.md              # 本文档
```

## 故障排除

### ffmpeg未安装

```bash
# Conda环境
conda install -c conda-forge ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### pydantic版本冲突

```bash
pip install 'pydantic>=1.10.0,<2.0.0' --force-reinstall
```

## 许可证

本项目基于[FunASR](https://github.com/modelscope/FunASR)，遵循其许可证。

## 贡献

欢迎提交Issue和Pull Request！

## 致谢

- [FunASR](https://github.com/modelscope/FunASR) - 阿里巴巴达摩院语音识别框架
- [FunAudioLLM](https://funaudiollm.github.io/) - 多语言模型
