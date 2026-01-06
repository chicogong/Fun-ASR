# Fun-ASR MLT 部署与使用指南

> 多语言语音识别服务 - 支持31种语言

---

## 🚀 快速开始

### 一键部署（推荐）

```bash
git clone https://github.com/chicogong/Fun-ASR.git
cd Fun-ASR
./docker-start.sh
```

或直接使用远程镜像：

```bash
docker pull ccr.ccs.tencentyun.com/chico/funasr-mlt-batch:latest

docker run -d --name funasr \
  --gpus all \
  -p 8000:8000 \
  -e USE_GPU=true \
  -e MAX_BATCH_SIZE=50 \
  -e NUM_WORKERS=12 \
  -v ~/.cache/modelscope:/root/.cache/modelscope \
  ccr.ccs.tencentyun.com/chico/funasr-mlt-batch:latest
```

### 本地部署

```bash
./run.sh local
```

---

## ✅ 验证部署

```bash
# 健康检查
curl http://localhost:8000/health

# 单文件识别
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=zh"

# 批量识别
curl -X POST http://localhost:8000/transcribe_batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "language=zh"

# API文档
浏览器访问: http://localhost:8000/docs
```

---

## ⚙️ 配置说明

### 环境变量

| 变量 | 默认值 | 说明 |
|------|-------|------|
| USE_GPU | true | GPU模式（推荐） |
| MODEL_PATH | FunAudioLLM/Fun-ASR-MLT-Nano-2512 | 模型路径 |
| MAX_BATCH_SIZE | 50 | 最大批量大小 |
| NUM_WORKERS | 12 | Worker数量 |

### 性能配置建议

| GPU | Workers | 预估日处理 |
|-----|---------|----------|
| Tesla T4 (15GB) | 4 | ~340h |
| Tesla L20 (44GB) | 12 | ~510h |
| A10 (24GB) | 6 | ~425h |

**注意**: Docker多worker模式可能不稳定，建议本地部署或使用单worker模式（`NUM_WORKERS=1`）

---

## 🌍 支持的语言

31种语言：中文、英文、日文、韩文、粤语、越南语、印尼语、泰语、马来语、菲律宾语、阿拉伯语、印地语、保加利亚语、克罗地亚语、捷克语、丹麦语、荷兰语、爱沙尼亚语、芬兰语、希腊语、匈牙利语、爱尔兰语、拉脱维亚语、立陶宛语、马耳他语、波兰语、葡萄牙语、罗马尼亚语、斯洛伐克语、斯洛文尼亚语、瑞典语

---

## 📊 性能说明

### MLT模型特点

| 特性 | 说明 |
|------|------|
| 多语言支持 | ✅ 31种语言 |
| GPU日处理 | ~500小时（L20 12 workers） |
| GPU利用率 | ~7%（受架构限制） |
| 适用场景 | 多语言混合场景 |

### 与Paraformer性能对比

**测试环境**: Tesla T4, 60秒中文音频

| 模型 | RTF | 日处理 | GPU利用率 | 批量并行 |
|------|-----|--------|----------|---------|
| MLT | 0.0487 | 493h | 7% | ❌ 串行 |
| Paraformer | 0.0069 | 3495h | 50%+ | ✅ 并行 |
| **性能差距** | **7.1x** | **7.1x** | **7x** | - |

### 为什么性能差距这么大？

**核心原因**: 架构差异

| | MLT | Paraformer |
|---|-----|-----------|
| 架构类型 | 自回归 (Autoregressive) | 非自回归 (Non-autoregressive) |
| 解码方式 | 逐个token预测，串行 | 所有token同时预测，并行 |
| 设计目标 | 支持多语言 | 中文高性能 |
| 批量处理 | 伪批量（循环） | 真批量（并行） |
| 多worker优化 | 提升有限 | 指数级提升 |

**简单类比**:
- **MLT**: 1个收银员，顾客排队结账（串行）
- **Paraformer**: 10个收银员，顾客同时结账（并行）

**技术细节**:
```python
# MLT (自回归) - 必须串行
for audio in batch:
    for i in range(max_len):
        token[i] = decoder(audio, token[0:i])  # 依赖前面的token

# Paraformer (非自回归) - 可以并行
batch_tokens = decoder(batch_audio)  # 一次性预测所有
```

---

## 🎯 部署建议

### 场景1: 纯多语言

**推荐**: MLT（当前部署）
- 日处理: ~500小时
- 成本: 单GPU
- 优势: 支持31种语言

### 场景2: 纯中文高性能

**推荐**: Paraformer
```bash
docker run -d --name funasr-para \
  --gpus all -p 8000:18001 \
  -e NUM_WORKERS=8 \
  -e MAX_BATCH_SIZE=20 \
  -e MODEL_PATH="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
  -v ~/.cache/modelscope:/root/.cache/modelscope \
  ccr.ccs.tencentyun.com/chico/funasr-nano-batch:latest
```
- 日处理: ~15,000-20,000小时
- 成本效益: MLT的7-25倍
- 限制: 仅支持中文

### 场景3: 中文为主+少量多语言

**推荐**: 混合部署
- 主服务: Paraformer (端口8000) - 处理中文
- 备用: MLT (端口8001) - 处理其他语言
- 路由: 根据语言分发请求

---

## 🔧 故障排除

### Docker部署问题

**多worker模式不稳定**:
```bash
# 解决: 使用单worker
-e NUM_WORKERS=1
```

**GPU crash**:
```bash
# 解决: 使用CPU模式
-e USE_GPU=false
```

**端口占用**:
```bash
# 检查并清理
lsof -i :8000
docker rm -f funasr
```

### 本地部署问题

**Python版本不兼容**:
```bash
# 需要Python 3.11
python3.11 --version
sudo apt install python3.11 python3.11-venv
```

**模型加载失败**:
```bash
# 确保FunASR版本正确
pip install "funasr==1.2.9" --force-reinstall
```

---

## 📝 API参考

### 单文件识别

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=zh" \
  -F "hotwords=语音,识别"
```

**响应**:
```json
{
  "text": "识别的文本内容",
  "language": "zh"
}
```

### 批量识别

```bash
curl -X POST http://localhost:8000/transcribe_batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "language=zh"
```

**响应**:
```json
{
  "results": [
    {"filename": "audio1.wav", "text": "...", "language": "zh"},
    {"filename": "audio2.wav", "text": "...", "language": "zh"}
  ]
}
```

---

## 💡 生产部署建议

1. **本地部署优先** - Docker GPU兼容性问题较多
2. **单worker稳定** - Docker多worker模式不稳定
3. **监控GPU** - `nvidia-smi` 监控使用率
4. **Nginx代理** - 负载均衡和SSL
5. **限流保护** - 防止服务过载

---

## 📚 更多信息

- GitHub: https://github.com/chicogong/Fun-ASR
- API文档: http://localhost:8000/docs
- 问题反馈: GitHub Issues
