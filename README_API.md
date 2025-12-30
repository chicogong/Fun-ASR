# Fun-ASR API 使用文档

基于 Fun-ASR-Nano 的语音识别 HTTP API 服务，支持双模型部署和批量推理。

## 快速开始

### 启动服务

```bash
# Docker 方式 (推荐)
docker-compose up -d

# 或本地启动
python server.py
```

### 测试服务

```bash
curl http://localhost:8000/health
# {"status":"ok","models_loaded":["nano","mlt"]}
```

---

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/info` | GET | 服务配置信息 |
| `/models` | GET | 查看已加载模型 |
| `/transcribe` | POST | 单文件识别 |
| `/transcribe_batch` | POST | 批量文件识别 |

---

## 详细接口说明

### 1. 健康检查

```bash
curl http://localhost:8000/health
```

**响应示例：**
```json
{
    "status": "ok",
    "models_loaded": ["nano", "mlt"]
}
```

---

### 2. 查看服务信息

```bash
curl http://localhost:8000/info
```

**响应示例：**
```json
{
    "models": {
        "nano": {
            "name": "Fun-ASR-Nano-2512",
            "loaded": true,
            "languages": ["中文", "英文", "日文", "粤语", "韩文"],
            "description": "中文优化模型"
        },
        "mlt": {
            "name": "Fun-ASR-MLT-Nano-2512",
            "loaded": true,
            "languages": ["zh", "en", "ja", "ko", "yue", ...],
            "description": "多语言模型 (31种语言)"
        }
    },
    "default_model": "nano",
    "max_batch_size": 20,
    "device": "cuda:0",
    "gpu_name": "Tesla T4",
    "gpu_memory_total_gb": 14.6,
    "gpu_memory_used_gb": 6.2
}
```

---

### 3. 查看已加载模型

```bash
curl http://localhost:8000/models
```

**响应示例：**
```json
{
    "available": ["nano", "mlt"],
    "default": "nano",
    "details": {
        "nano": {
            "name": "Fun-ASR-Nano-2512",
            "description": "中文优化模型",
            "languages": ["中文", "英文", "日文", "粤语", "韩文"]
        },
        "mlt": {
            "name": "Fun-ASR-MLT-Nano-2512",
            "description": "多语言模型 (31种语言)",
            "languages": ["zh", "en", "ja", "ko", "yue", "vi", "id", "th", "ms", "tl"]
        }
    }
}
```

---

### 4. 单文件识别

**请求：**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "model=nano" \
  -F "language=中文" \
  -F "hotwords=关键词1,关键词2" \
  -F "itn=true"
```

**参数说明：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| file | File | 是 | - | 音频文件 (wav/mp3/flac等) |
| model | string | 否 | nano | 模型选择: `nano` 或 `mlt` |
| language | string | 否 | auto | 语言代码 (见下表) |
| hotwords | string | 否 | "" | 热词，逗号分隔 |
| itn | bool | 否 | true | 是否进行文本规整 |

**响应示例：**
```json
{
    "text": "嗯，比如我去日本的时候，他就会感受到一股特别浓厚的中国文化。",
    "model": "nano"
}
```

---

### 5. 批量文件识别

**请求：**
```bash
curl -X POST http://localhost:8000/transcribe_batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav" \
  -F "model=nano" \
  -F "language=中文"
```

**参数说明：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| files | File[] | 是 | - | 多个音频文件 (最多20个) |
| model | string | 否 | nano | 模型选择 |
| language | string | 否 | auto | 语言代码 |
| hotwords | string | 否 | "" | 热词 |
| itn | bool | 否 | true | 文本规整 |

**响应示例：**
```json
{
    "results": [
        {"filename": "audio1.wav", "text": "第一段识别结果..."},
        {"filename": "audio2.wav", "text": "第二段识别结果..."},
        {"filename": "audio3.wav", "text": "第三段识别结果..."}
    ],
    "model": "nano",
    "batch_size": 3
}
```

---

## 语言代码

### Nano 模型 (中文优化)

| 语言 | 代码 |
|------|------|
| 中文 | `中文` |
| 英文 | `英文` |
| 日文 | `日文` |
| 韩文 | `韩文` |
| 粤语 | `粤语` |

### MLT 模型 (31种语言)

| 语言 | 代码 | 语言 | 代码 |
|------|------|------|------|
| 中文 | zh | 英文 | en |
| 日文 | ja | 韩文 | ko |
| 粤语 | yue | 越南语 | vi |
| 印尼语 | id | 泰语 | th |
| 马来语 | ms | 菲律宾语 | tl |
| 阿拉伯语 | ar | 印地语 | hi |
| 保加利亚语 | bg | 克罗地亚语 | hr |
| 捷克语 | cs | 丹麦语 | da |
| 荷兰语 | nl | 爱沙尼亚语 | et |
| 芬兰语 | fi | 希腊语 | el |
| 匈牙利语 | hu | 爱尔兰语 | ga |
| 拉脱维亚语 | lv | 立陶宛语 | lt |
| 马耳他语 | mt | 波兰语 | pl |
| 葡萄牙语 | pt | 罗马尼亚语 | ro |
| 斯洛伐克语 | sk | 斯洛文尼亚语 | sl |
| 瑞典语 | sv | | |

---

## 代码示例

### Python

```python
import requests

# 单文件识别
def transcribe(file_path, model="nano", language="中文"):
    with open(file_path, 'rb') as f:
        response = requests.post(
            'http://localhost:8000/transcribe',
            files={'file': f},
            data={'model': model, 'language': language}
        )
    return response.json()

# 使用示例
result = transcribe('audio.wav')
print(result['text'])

# 批量识别
def transcribe_batch(file_paths, model="nano", language="中文"):
    files = [('files', open(fp, 'rb')) for fp in file_paths]
    response = requests.post(
        'http://localhost:8000/transcribe_batch',
        files=files,
        data={'model': model, 'language': language}
    )
    # 关闭文件
    for _, f in files:
        f.close()
    return response.json()

# 使用示例
results = transcribe_batch(['audio1.wav', 'audio2.wav', 'audio3.wav'])
for r in results['results']:
    print(f"{r['filename']}: {r['text']}")
```

### JavaScript (Node.js)

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function transcribe(filePath, model = 'nano', language = '中文') {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    form.append('model', model);
    form.append('language', language);
    
    const response = await axios.post('http://localhost:8000/transcribe', form, {
        headers: form.getHeaders()
    });
    return response.data;
}

// 使用示例
transcribe('audio.wav').then(result => {
    console.log(result.text);
});
```

### cURL

```bash
# 单文件识别
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "model=nano" \
  -F "language=中文"

# 批量识别
curl -X POST http://localhost:8000/transcribe_batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "model=nano"

# 使用 MLT 模型识别英文
curl -X POST http://localhost:8000/transcribe \
  -F "file=@english.wav" \
  -F "model=mlt" \
  -F "language=en"
```

---

## 错误处理

### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求错误 (空文件、格式不支持等) |
| 503 | 模型未加载 |

### 错误响应格式

```json
{
    "detail": "错误描述信息"
}
```

### 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| Empty file | 上传了空文件 | 检查文件是否正确 |
| Model not loaded | 模型未加载 | 等待服务启动完成 |
| Too many files | 超过最大 batch 数 | 减少文件数量或分批请求 |

---

## 性能说明

### 吞吐量 (Tesla T4 15GB)

| 模式 | 吞吐量 | 延迟 | 实时率 |
|------|--------|------|--------|
| 单文件 | ~1.3 files/s | ~0.8s | 8.9x |
| Batch=20 | ~10 files/s | ~2s | 61x |

> 最优 batch size 为 20，吞吐量约 10 files/s

### 显存占用

| 配置 | 显存 |
|------|------|
| Nano 单模型 | ~3.4GB |
| MLT 单模型 | ~3.4GB |
| 双模型 | ~4.2GB |

### 性能测试

```bash
# 运行性能对比测试
python tests/test_performance.py --audio-dir /path/to/audio

# 查找最优 batch size
python tests/test_performance.py --find-optimal
```

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| MODEL_NANO_PATH | HuggingFace ID | Nano 模型路径 |
| MODEL_MLT_PATH | HuggingFace ID | MLT 模型路径 |
| LOAD_NANO | true | 是否加载 Nano |
| LOAD_MLT | true | 是否加载 MLT |
| DEFAULT_MODEL | nano | 默认模型 |
| MAX_BATCH_SIZE | 20 | 最大批处理数 |
