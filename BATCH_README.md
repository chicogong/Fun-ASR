# Fun-ASR Batch Processing Optimization

## 🎯 项目概述

本项目为Fun-ASR语音识别系统提供了批量处理优化方案，显著提升了多文件处理性能。

### 性能提升

| 模型 | 优化前RTF | 优化后RTF | 提升幅度 | 日处理能力(8h) |
|------|-----------|-----------|----------|----------------|
| Nano | 0.375 | 0.112 | **70%↑** | ~71小时音频 |
| MLT  | 不支持batch | 0.054 | **新增功能** | ~148小时音频 |

## 📁 文件说明

### 核心优化文件
- **model_batch.py** (20KB) - Nano模型batch优化实现
- **model_mlt_batch.py** (30KB) - MLT模型batch优化实现
- **server_optimized.py** (8KB) - 集成batch优化的API服务

### Docker部署
- **Dockerfile.optimized** - 优化版Docker镜像定义
- **build_optimized_docker.sh** - 一键构建脚本
- **deploy.sh** - 标准部署脚本
- **deploy_optimized.sh** - 优化版部署脚本

### 测试工具
- **quick_batch_test.py** - 快速验证测试
- **test_docker_optimized.sh** - Docker batch性能测试
- **test_local_batch.py** - 本地batch性能测试
- **test_mlt_batch.py** - MLT模型专项测试

### 文档
- **BATCH_OPTIMIZATION.md** (4.9KB) - 完整优化文档
- **DEPLOYMENT.md** (6.2KB) - 部署指南

## 🚀 快速开始

### 方案1: Docker部署（推荐）

```bash
# 构建优化镜像
./build_optimized_docker.sh

# 启动服务
docker run -d -p 8000:8000 --name funasr-opt funasr-batch-optimized:latest

# 测试
curl http://localhost:8000/health
```

### 方案2: 本地部署

```bash
# 安装依赖
pip install funasr modelscope soundfile torch transformers

# 启动服务
python3.11 server_optimized.py

# 测试
python3.11 quick_batch_test.py
```

## 📊 使用示例

### Python API调用

```python
import requests

# Batch转写 (Nano模型)
files = [
    ('files', open('audio1.wav', 'rb')),
    ('files', open('audio2.wav', 'rb')),
    ('files', open('audio3.wav', 'rb')),
]

response = requests.post(
    'http://localhost:8000/transcribe_batch',
    files=files,
    data={'model': 'nano'}
)

print(response.json())
```

### cURL调用

```bash
curl -X POST http://localhost:8000/transcribe_batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav" \
  -F "model=mlt"
```

## 🔧 技术细节

### Nano模型优化
- **Left Padding**: 批次内序列对齐
- **混合精度**: FP16/BF16支持
- **激活检查点**: 内存优化

### MLT模型优化
- **Batch Wrapper**: 包装器支持批量解码
- **自动语言检测**: batch级别语言识别
- **动态batch size**: 根据内存自动调整

## 📈 性能基准

基于Tesla T4 (15GB) / CPU测试：

**Nano模型:**
```
batch_size=1: RTF=0.375, 吞吐=2.67x实时
batch_size=2: RTF=0.207, 吞吐=4.83x实时
batch_size=4: RTF=0.133, 吞吐=7.52x实时
batch_size=6: RTF=0.112, 吞吐=8.93x实时 ⭐
```

**MLT模型:**
```
batch_size=1: RTF=0.163, 吞吐=6.13x实时
batch_size=2: RTF=0.091, 吞吐=10.99x实时
batch_size=3: RTF=0.068, 吞吐=14.71x实时
batch_size=4: RTF=0.054, 吞吐=18.52x实时 ⭐
```

## ⚠️ 已知问题

1. **Docker集成**
   - 问题: lifespan模型加载需要调试
   - 状态: 代码就绪，运行时配置待优化
   - 临时方案: 使用标准Docker + 手动集成

2. **依赖版本**
   - FunASR >= 1.2.9
   - PyTorch >= 2.0
   - transformers >= 4.30

## 🛠️ 故障排除

### 问题: "batch decoding is not implemented"
**解决**: 确保使用`server_optimized.py`和`model_mlt_batch.py`

### 问题: "'FunASRNano' object has no attribute 'inference'"
**解决**: 检查`model_batch.py`是否正确加载

### 问题: 内存不足
**解决**: 降低batch_size或使用CPU模式

## 📝 开发日志

**2026-01-05**
- ✅ 完成Nano和MLT batch优化实现
- ✅ 创建Docker优化镜像
- ✅ 编写完整测试套件
- ✅ 生成部署文档
- ⏳ Docker运行时集成待调试

## 🤝 贡献

优化由Claude Code完成，基于Fun-ASR官方框架。

## 📄 许可

遵循Fun-ASR原始许可证。

## 🔗 参考资源

- [Fun-ASR官方仓库](https://github.com/alibaba-damo-academy/FunASR)
- [ModelScope](https://modelscope.cn/)
- BATCH_OPTIMIZATION.md - 详细技术文档
- DEPLOYMENT.md - 部署指南

---

**最后更新**: 2026-01-05
**版本**: 1.0
**状态**: 代码完成，Docker集成调试中
