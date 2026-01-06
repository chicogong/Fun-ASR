# Fun-ASR MLT Batch Server 测试

## 📁 目录结构

```
tests/
├── download_multilingual_test_data.py  # 下载多语言测试数据
├── test_multilingual_batch.py          # 测试多语言性能
└── README.md                            # 本文件
```

## 🚀 快速开始

### 1. 下载测试数据

```bash
# 下载多语言测试样本（10种语言，每种5个样本）
python tests/download_multilingual_test_data.py
```

这会下载以下语言的测试数据：
- 中文、英语、日语、韩语
- 法语、德语、西班牙语、俄语
- 阿拉伯语、印地语

### 2. 启动Batch Server

```bash
# 本地模式
./start_batch_server.sh local

# Docker模式
./start_batch_server.sh docker
```

### 3. 运行多语言性能测试

```bash
# 测试所有语言的batch性能
python tests/test_multilingual_batch.py
```

## 📊 性能指标

测试会输出：
- 每种语言的RTF (Real-Time Factor)
- 处理速度
- 识别结果示例
- 每天处理能力估算

## 💡 其他测试

### HTTP API测试

```bash
# 测试健康检查
curl http://localhost:8000/health

# 批量转录
curl -X POST http://localhost:8000/transcribe_batch \
  -F "files=@test_data/multilingual/中文/zh_cn_0.wav" \
  -F "files=@test_data/multilingual/英语/en_us_0.wav"

# 查看统计
curl http://localhost:8000/stats
```

## 📈 预期结果

基于batch_size=6的测试：
- RTF: ~0.03
- 处理速度: ~30x 实时
- 每天处理: ~700-800小时音频
