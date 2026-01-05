# Fun-ASR 部署指南

Fun-ASR是一个端到端的大规模语音识别模型，支持31种语言的高精度语音转文字。本文档提供完整的部署和测试指南。

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (可选，用于GPU加速)
- 8GB+ RAM
- 20GB+ 存储空间

### 一键部署

```bash
# 克隆仓库
git clone https://github.com/FunAudioLLM/Fun-ASR.git
cd Fun-ASR

# 运行部署脚本
chmod +x deploy.sh
./deploy.sh
```

### 手动部署

```bash
# 1. 安装依赖
pip install -r requirements.txt
pip install gradio soundfile jiwer pandas jieba

# 2. 启动Web服务
python3.8 app.py
```

## 📖 功能特性

### 🌍 多语言支持

| 语言类别 | 支持语言 | 说明 |
|---------|---------|------|
| **主要语言** | 中文、英文、日文 | Fun-ASR-Nano-2512 |
| **扩展语言** | 31种语言 | Fun-ASR-MLT-Nano-2512 |
| **中文方言** | 7种方言 | 吴、粤、闽、客、赣、湘、晋 |
| **地方口音** | 26种口音 | 河南、陕西、湖北、四川等 |

### 🎯 核心能力

- **高精度识别**: 远场高噪声环境准确率达93%
- **实时转录**: 支持低延迟实时语音识别
- **音乐背景**: 支持音乐背景下的歌词识别
- **专业术语**: 擅长教育、金融等垂直领域

## 🏗️ 部署架构

### Web服务架构

```
Fun-ASR Web Service
├── app.py                    # Gradio Web界面
├── model.py                  # 模型核心代码
├── evaluation/               # 评测工具
│   ├── wer_calculator.py    # WER/CER计算
│   └── benchmark_runner.py  # 基准测试
├── data/                    # 测试数据
└── evaluation/results/      # 评测结果
```

### Docker部署

#### CPU版本
```bash
docker-compose up --build -d fun-asr
```

#### GPU版本
```bash
docker-compose --profile gpu up --build -d fun-asr-gpu
```

## 🔧 配置说明

### 模型配置

```python
# 基础配置
model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"  # 模型路径
device = "cuda:0"  # 设备选择: cuda:0, cpu, mps
language = "中文"   # 识别语言

# 高级配置
batch_size = 1     # 批处理大小
itn = True         # 逆文本规范化
hotwords = []      # 热词列表
```

### Web服务配置

```python
# Gradio配置
server_name = "0.0.0.0"  # 服务地址
server_port = 7860       # 服务端口
share = False           # 是否创建公共链接
```

## 📊 性能测试

### 基准测试

```bash
# 运行完整基准测试
python evaluation/benchmark_runner.py --config benchmark_config.json

# 创建测试配置
python evaluation/benchmark_runner.py --create-config
```

### WER计算

```bash
# 文件对比模式
python evaluation/wer_calculator.py \
    --mode file \
    --reference reference.txt \
    --hypothesis hypothesis.txt \
    --language zh \
    --output results.csv

# JSON数据集模式
python evaluation/wer_calculator.py \
    --mode json \
    --dataset dataset.json \
    --output results.csv
```

### 基本功能测试

```bash
# 运行环境和模型测试
python test_basic.py

# 手动下载模型
python download_model.py
```

## 🌐 Web界面使用

### 访问地址

- 本地访问: http://localhost:7860
- 网络访问: http://YOUR_IP:7860

### 功能说明

1. **音频上传**: 支持MP3、WAV、M4A等格式
2. **实时录音**: 直接使用麦克风录制
3. **语音识别**: 一键转换语音为文字
4. **结果展示**: 显示识别结果和处理时间
5. **系统监控**: 查看设备和模型状态

### 支持的音频格式

| 格式 | 扩展名 | 说明 |
|-----|-------|------|
| MP3 | .mp3 | 常用压缩格式 |
| WAV | .wav | 无损音频格式 |
| M4A | .m4a | Apple音频格式 |
| FLAC | .flac | 无损压缩格式 |

## 🚨 故障排除

### 常见问题

#### 1. 模型下载失败
```
错误: FunAudioLLM/Fun-ASR-Nano-2512 is not registered
```

**解决方案:**
- 检查网络连接
- 运行 `python download_model.py` 手动下载
- 等待模型下载完成后重试

#### 2. CUDA内存不足
```
错误: CUDA out of memory
```

**解决方案:**
- 使用CPU模式: `device="cpu"`
- 减少batch_size
- 清理GPU内存: `torch.cuda.empty_cache()`

#### 3. 音频格式不支持
```
错误: 无法读取音频文件
```

**解决方案:**
- 使用支持的格式 (MP3/WAV/M4A)
- 检查音频文件是否完整
- 转换音频格式

### 性能优化

#### GPU加速
```python
# 确保使用GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 优化内存使用
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
```

#### 批处理优化
```python
# 批量处理多个文件
batch_size = 4  # 根据GPU内存调整
```

## 📈 基准测试结果

### 开源数据集性能 (WER %)

| 测试集 | Fun-ASR-nano | Whisper-large-v3 | 说明 |
|-------|--------------|------------------|------|
| AIShell1 | **1.80** | 4.72 | 中文识别 |
| Fleurs-zh | **2.56** | 5.18 | 中文多样性 |
| Fleurs-en | **5.96** | 6.23 | 英文识别 |
| Librispeech-clean | **1.76** | 1.86 | 英文清晰语音 |

### 工业数据集性能 (WER %)

| 场景 | Fun-ASR-nano | 说明 |
|-----|-------------|------|
| 近场识别 | **7.79** | 清晰环境 |
| 远场识别 | **5.79** | 会议室等 |
| 复杂背景 | **14.59** | 噪声环境 |
| 方言识别 | **28.18** | 中文方言 |
| 口音识别 | **12.90** | 地方口音 |

## 🔗 相关资源

### 官方资源
- [项目主页](https://funaudiollm.github.io/funasr)
- [ModelScope](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512)
- [Hugging Face](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)

### 在线体验
- [ModelScope空间](https://modelscope.cn/studios/FunAudioLLM/Fun-ASR-Nano)
- [Hugging Face空间](https://huggingface.co/spaces/FunAudioLLM/Fun-ASR-Nano)

### 技术文档
- [技术报告](https://arxiv.org/abs/2509.12508)
- [微调文档](docs/finetune.md)
- [FunASR工具包](https://github.com/modelscope/FunASR)

## 📞 支持与反馈

如果您在部署过程中遇到问题，请：

1. 查看本文档的故障排除部分
2. 检查 [Issues页面](https://github.com/FunAudioLLM/Fun-ASR/issues)
3. 提交新的Issue描述问题

---

**注意**: 首次运行需要下载模型文件（约800MB），请确保网络连接稳定。