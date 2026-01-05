# Fun-ASR-Nano 快速参考卡片

## 核心组件一览

| 组件 | 类型 | 输入维度 | 输出维度 | 状态 |
|------|------|----------|----------|------|
| **WavFrontend** | 特征提取 | 16kHz 波形 | [B, T, 80] Fbank | - |
| **Audio Encoder** | SenseVoiceEncoderSmall | [B, T, 80] | [B, T', 512] | Frozen |
| **Audio Adaptor** | Transformer (2层) | [B, T', 512] | [B, T'', 1024] | Frozen |
| **LLM** | Qwen3-0.6B | [B, seq, 1024] | [B, seq, vocab] | Frozen |
| **Tokenizer** | Qwen3 | 文本 | Token IDs | - |

## 关键方法调用链

```
inference()
    └── inference_llm()
            └── inference_prepare()
                    ├── data_template()      # 解析对话格式
                    ├── data_load_speech()   # 加载音频+构建序列
                    ├── encode()             # 音频编码
                    └── audio_adaptor()      # 维度对齐
            └── llm.generate()               # 文本生成
            └── tokenizer.decode()           # 解码输出
```

## 特殊标记说明

| 标记 | 含义 |
|------|------|
| `<\|im_start\|>` | ChatML 对话开始 |
| `<\|im_end\|>` | ChatML 对话结束 |
| `<\|startofspeech\|>` | 语音内容开始 |
| `<\|endofspeech\|>` | 语音内容结束 |
| `!path` | 音频文件路径 |
| `!!` | 使用 audio 字段中的 Tensor |

## 数据字典结构

```python
# data_load_speech() 输出
{
    "speech":          [1, T, 80],      # Fbank特征
    "speech_lengths":  [1] or [1, turns],
    "input_ids":       [1, seq_len],    # 完整token序列
    "attention_mask":  [1, seq_len],    # 全1
    "labels_ids":      [seq_len],       # 源=-100, 目标=token_id
    "fbank_beg":       [1, turns],      # 语音起始位置
    "fake_token_len":  [1, turns],      # 语音token长度
    "source_ids":      [1, src_len],    # 仅输入部分
    "target_ids":      [1, tgt_len],    # 仅输出部分
}
```

## 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hotwords` | [] | 热词列表，提高特定词识别率 |
| `language` | None | 目标语言 (中文/英文/日文等) |
| `itn` | True | 是否进行文本规整 |
| `max_length` | 512 | 最大生成长度 |
| `batch_size` | 1 | 批量大小 |
| `fp16` | False | 使用FP16推理 |
| `bf16` | False | 使用BF16推理 |

## 快速使用示例

```python
from funasr import AutoModel

# 加载模型
model = AutoModel(
    model="/path/to/Fun-ASR-Nano-2512",
    trust_remote_code=True,
    remote_code="./model.py",
    device="cuda:0"  # 或 "cpu"
)

# 基本推理
result = model.generate(
    input=["audio.mp3"],
    cache={},
    batch_size=1
)
print(result[0]["text"])

# 带热词和语言设置
result = model.generate(
    input=["audio.mp3"],
    cache={},
    hotwords=["开放时间", "北京"],
    language="中文",
    itn=True
)
```

## 维度变化追踪

```
音频 (16kHz)
    ↓ WavFrontend (frame_shift=10ms, lfr_n=6)
[B, samples] → [B, T, 80]  (T ≈ samples/160/6)
    ↓ Audio Encoder (50层 SANM)
[B, T, 80] → [B, T', 512]  (T' = T)
    ↓ Audio Adaptor (2层 Transformer, low_frame_rate)
[B, T', 512] → [B, T'', 1024]  (T'' ≈ T'/4)
    ↓ 融合到 LLM embedding
[B, seq_len, 1024]
    ↓ Qwen3-0.6B (28层)
[B, seq_len, vocab_size]
    ↓ argmax + decode
文本输出
```

## 训练关键配置

```yaml
# 优化器
optim: adamw
optim_conf:
  lr: 5.0e-06
  weight_decay: 0.0

# 学习率调度
scheduler: warmuplr
scheduler_conf:
  warmup_steps: 2500

# 训练设置
train_conf:
  accum_grad: 1
  grad_clip: 5
  max_epoch: 2
  use_deepspeed: true
  use_bf16: false
```

## 文件结构

```
Fun-ASR-Nano-2512/
├── config.yaml          # 模型配置
├── configuration.json   # 模型元信息
├── model.pt            # 模型权重 (~2GB)
├── multilingual.tiktoken
├── Qwen3-0.6B/         # LLM配置
│   ├── config.json
│   ├── tokenizer.json
│   └── vocab.json
└── example/            # 示例音频
    ├── zh.mp3
    ├── en.mp3
    └── ja.mp3
```

## 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `'type' object is not subscriptable` | Python 3.8 与 modelscope 不兼容 | 使用 Python 3.9+ |
| `qwen3 not recognized` | transformers 版本过旧 | 升级到 4.47+ |
| `No space left on device` | 磁盘空间不足 | 清理缓存或挂载新磁盘 |
| `'str' object has no attribute 'size'` | 音频加载失败 | 检查音频路径和依赖 |
