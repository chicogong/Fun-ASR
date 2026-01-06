# Fun-ASR MLT Batch Server - 快速启动指南

## 🚀 一键启动（推荐）

### Linux/Mac

```bash
# 方法1: 使用启动脚本（自动安装所有依赖）
./start_batch_server.sh local
```

脚本会自动：
1. ✅ 检测Python版本
2. ✅ 创建/激活虚拟环境
3. ✅ 安装所有依赖
4. ✅ 启动服务

### 如果遇到问题

```bash
# 手动安装依赖
pip install -r requirements-batch-server.txt

# 然后启动服务
python server_batch.py
```

## 🔧 手动安装步骤

### 1. 创建虚拟环境

```bash
# Python 3.8+
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

### 2. 安装依赖

```bash
# 升级pip
pip install --upgrade pip

# 安装所有依赖
pip install -r requirements-batch-server.txt
```

### 3. 启动服务

```bash
python server_batch.py
```

## 📋 依赖说明

### 核心依赖
- **FastAPI** >= 0.95.0 - Web框架
- **Uvicorn** >= 0.20.0 - ASGI服务器
- **python-multipart** - 文件上传支持

### FunASR依赖
- **funasr** >= 1.3.0 - FunASR核心库
- **torch** >= 2.0.0 - PyTorch
- **transformers** >= 4.30.0 - Transformers库

### 测试依赖（可选）
- **datasets** >= 2.14.0 - 多语言测试数据
- **soundfile** >= 0.12.0 - 音频处理

## 🐳 Docker部署

```bash
# 一键启动Docker
./start_batch_server.sh docker

# 手动构建和运行
docker build -t funasr-mlt-batch:latest -f Dockerfile.batch .
docker run -d --name funasr-batch --gpus all -p 8000:8000 funasr-mlt-batch:latest
```

## ✅ 验证安装

```bash
# 健康检查
curl http://localhost:8000/health

# 查看性能统计
curl http://localhost:8000/stats

# API文档
浏览器打开: http://localhost:8000/docs
```

## 🐛 常见问题

### 1. ModuleNotFoundError: No module named 'fastapi'

**解决**:
```bash
pip install fastapi uvicorn python-multipart
```

### 2. 虚拟环境未激活

**解决**:
```bash
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

### 3. CUDA相关错误

**解决**:
- 确保安装了GPU版本的PyTorch
- 检查CUDA版本兼容性
- 可以使用CPU模式: `DEVICE="cpu" python server_batch.py`

### 4. 端口8000被占用

**解决**:
```bash
# 使用其他端口
python server_batch.py --port 8001
```

## 📞 获取帮助

- 查看完整文档: [BATCH_SERVER_README.md](BATCH_SERVER_README.md)
- 测试指南: [tests/README.md](tests/README.md)
- 部署文档: [DEPLOY.md](DEPLOY.md)

## 🎯 快速测试

```bash
# 测试单文件转录
curl -X POST http://localhost:8000/transcribe \
  -F "file=@test.wav"

# 测试批量转录（推荐6个文件）
curl -X POST http://localhost:8000/transcribe_batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav" \
  -F "files=@audio4.wav" \
  -F "files=@audio5.wav" \
  -F "files=@audio6.wav"
```

---

**安装完成后，服务将在 http://localhost:8000 启动** 🚀
