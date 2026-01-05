# 本地部署 Fun-ASR

## 快速开始

```bash
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
