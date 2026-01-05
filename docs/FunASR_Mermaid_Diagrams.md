# Fun-ASR-Nano Mermaid 流程图

## 1. 整体架构图

```mermaid
flowchart TB
    subgraph Input["输入层"]
        A[音频文件/Tensor]
        B[文本 Prompt]
    end
    
    subgraph Frontend["前端处理"]
        C[WavFrontend<br/>16kHz, 80 Mel]
    end
    
    subgraph AudioPipeline["音频处理管线"]
        D[Audio Encoder<br/>SenseVoiceEncoderSmall<br/>50层, 512维]
        E[Audio Adaptor<br/>Transformer 2层<br/>512→1024维]
    end
    
    subgraph TextPipeline["文本处理管线"]
        F[Tokenizer<br/>Qwen3]
        G[Embedding Layer<br/>1024维]
    end
    
    subgraph Fusion["特征融合"]
        H[Embedding 替换<br/>将音频特征插入<br/>文本embedding序列]
    end
    
    subgraph LLM["大语言模型"]
        I[Qwen3-0.6B<br/>28层 Transformer<br/>1024维]
    end
    
    subgraph Output["输出层"]
        J[Tokenizer Decode]
        K[识别文本]
    end
    
    A --> C
    C --> D
    D --> E
    E --> H
    
    B --> F
    F --> G
    G --> H
    
    H --> I
    I --> J
    J --> K
```

## 2. 数据处理流程图

```mermaid
flowchart TD
    subgraph Step1["Step 1: 输入处理"]
        A1[原始输入<br/>音频路径/Tensor] --> A2[inference方法]
        A2 --> A3[构建Prompt<br/>热词+语言+ITN设置]
        A3 --> A4[构建对话格式<br/>system/user/assistant]
    end
    
    subgraph Step2["Step 2: 模板转换"]
        B1[data_template方法] --> B2[解析角色]
        B2 --> B3{角色类型?}
        B3 -->|system| B4[system列表]
        B3 -->|user| B5[user列表<br/>包含音频标记]
        B3 -->|assistant| B6[assistant列表]
    end
    
    subgraph Step3["Step 3: 数据加载"]
        C1[data_load_speech方法]
        C2[构建ChatML格式<br/>im_start/im_end标记]
        C3[正则分割<br/>分离文本和语音标记]
        C4{是语音标记?}
        C5[tokenizer编码<br/>生成token ids]
        C6[加载音频文件<br/>load_audio_text_image_video]
        C7[提取Fbank特征<br/>extract_fbank]
        C8[计算fake_token_len<br/>预估encoder输出长度]
        C9[创建占位token<br/>fake_token = 0 * len]
        
        C1 --> C2 --> C3 --> C4
        C4 -->|否| C5
        C4 -->|是| C6
        C6 --> C7 --> C8 --> C9
    end
    
    subgraph Step4["Step 4: 输出数据"]
        D1[speech: Fbank特征]
        D2[input_ids: Token序列]
        D3[fbank_beg: 语音位置]
        D4[fake_token_len: 语音长度]
        D5[labels_ids: 训练标签]
    end
    
    A4 --> B1
    B4 & B5 & B6 --> C1
    C5 --> D2
    C9 --> D2
    C7 --> D1
    C8 --> D4
    C9 --> D3
```

## 3. 推理流程详细图

```mermaid
flowchart TD
    subgraph Inference["推理流程"]
        I1[inference入口] --> I2[构建prompt和对话格式]
        I2 --> I3[inference_llm]
        I3 --> I4[inference_prepare]
        
        subgraph Prepare["数据准备"]
            P1[data_template] --> P2[data_load_speech]
            P2 --> P3[to_device移至GPU]
        end
        
        I4 --> P1
        P3 --> I5{有音频?}
        
        subgraph AudioEncode["音频编码"]
            AE1[self.encode<br/>Audio Encoder] --> AE2[self.audio_adaptor<br/>维度对齐]
        end
        
        I5 -->|是| AE1
        AE2 --> I6[获取text embedding<br/>llm.get_input_embeddings]
        I5 -->|否| I6
        
        I6 --> I7[替换音频位置embedding<br/>inputs_embeds替换]
        
        subgraph Generate["LLM生成"]
            G1[llm.generate<br/>自回归生成]
            G2[tokenizer.batch_decode<br/>解码为文本]
            G3[后处理<br/>清理特殊字符]
        end
        
        I7 --> G1 --> G2 --> G3
        G3 --> I8[返回结果]
    end
```

## 4. Embedding 融合流程图

```mermaid
flowchart LR
    subgraph TextEmb["文本Embedding"]
        T1["[E_sys, E_user, E_0, E_0, E_0, E_assistant]"]
        T2["占位符位置: fbank_beg=2, len=3"]
    end
    
    subgraph AudioEmb["音频Embedding"]
        A1["Audio Encoder<br/>[1, T, 80] → [1, T', 512]"]
        A2["Audio Adaptor<br/>[1, T', 512] → [1, T'', 1024]"]
        A3["[A_0, A_1, A_2]<br/>1024维"]
    end
    
    subgraph Fusion["融合操作"]
        F1["inputs_embeds[0, 2:5] = audio_emb"]
    end
    
    subgraph Result["融合结果"]
        R1["[E_sys, E_user, A_0, A_1, A_2, E_assistant]"]
    end
    
    T1 --> Fusion
    A1 --> A2 --> A3
    A3 --> Fusion
    Fusion --> R1
```

## 5. 训练Forward流程图

```mermaid
flowchart TD
    subgraph Input["训练输入 Batch"]
        I1[speech: B x T x 80]
        I2[input_ids: B x seq_len]
        I3[labels_ids: B x seq_len]
        I4[fbank_beg, fake_token_len]
    end
    
    subgraph TextProcess["文本处理"]
        T1[input_ids过滤负值]
        T2[LLM embedding layer]
        T3[inputs_embeds: B x seq_len x 1024]
    end
    
    subgraph AudioProcess["音频处理"]
        A1{activation checkpoint?}
        A2[checkpoint包装encode]
        A3[直接encode]
        A4[audio_encoder<br/>B_speech x T x 80 → B_speech x T' x 512]
        A5[audio_adaptor<br/>B_speech x T' x 512 → B_speech x T'' x 1024]
    end
    
    subgraph Fusion["特征融合"]
        F1[遍历batch和turns]
        F2[根据fbank_beg替换embedding]
        F3[融合后的inputs_embeds]
    end
    
    subgraph LLMForward["LLM Forward"]
        L1[torch.autocast bf16]
        L2[labels处理: -1 → -100]
        L3[llm forward]
        L4[loss = CrossEntropy]
    end
    
    subgraph Stats["统计信息"]
        S1[compute_accuracy]
        S2[batch统计信息]
        S3[返回 loss, stats, weight]
    end
    
    I1 --> A1
    A1 -->|是| A2 --> A4
    A1 -->|否| A3 --> A4
    A4 --> A5
    
    I2 --> T1 --> T2 --> T3
    
    T3 --> F1
    A5 --> F1
    I4 --> F1
    F1 --> F2 --> F3
    
    F3 --> L1
    I3 --> L2
    L1 --> L3
    L2 --> L3
    L3 --> L4
    
    L4 --> S1
    L3 --> S1
    S1 --> S2 --> S3
```

## 6. 模型组件关系图

```mermaid
classDiagram
    class FunASRNano {
        +audio_encoder: SenseVoiceEncoderSmall
        +audio_adaptor: Transformer
        +llm: Qwen3
        +llm_dtype: str
        +use_low_frame_rate: bool
        +forward()
        +encode()
        +inference()
        +inference_prepare()
        +inference_llm()
        +data_template()
        +data_load_speech()
        +from_pretrained()
    }
    
    class SenseVoiceEncoderSmall {
        +output_size: 512
        +attention_heads: 4
        +num_blocks: 50
        +frozen: True
        +forward(speech, lengths)
    }
    
    class TransformerAdaptor {
        +encoder_dim: 512
        +llm_dim: 1024
        +n_layer: 2
        +use_low_frame_rate: True
        +forward(encoder_out, lengths)
    }
    
    class Qwen3 {
        +hidden_size: 1024
        +num_layers: 28
        +dtype: bf16
        +frozen: True
        +get_input_embeddings()
        +generate()
        +forward()
    }
    
    class WavFrontend {
        +fs: 16000
        +n_mels: 80
        +frame_length: 25
        +frame_shift: 10
        +lfr_m: 7
        +lfr_n: 6
        +forward(waveform)
    }
    
    class Tokenizer {
        +vocab_size: ~150000
        +encode(text)
        +decode(ids)
        +batch_decode(ids)
    }
    
    FunASRNano --> SenseVoiceEncoderSmall : audio_encoder
    FunASRNano --> TransformerAdaptor : audio_adaptor
    FunASRNano --> Qwen3 : llm
    FunASRNano ..> WavFrontend : uses
    FunASRNano ..> Tokenizer : uses
```

## 7. 数据流时序图

```mermaid
sequenceDiagram
    participant User
    participant Inference as inference()
    participant Template as data_template()
    participant LoadSpeech as data_load_speech()
    participant Frontend as WavFrontend
    participant Encoder as AudioEncoder
    participant Adaptor as AudioAdaptor
    participant LLM as Qwen3
    participant Tokenizer
    
    User->>Inference: 输入音频路径
    Inference->>Inference: 构建prompt
    Inference->>Template: 对话数据
    Template-->>Inference: 结构化contents
    
    Inference->>LoadSpeech: contents
    LoadSpeech->>Tokenizer: 文本部分encode
    Tokenizer-->>LoadSpeech: token ids
    LoadSpeech->>Frontend: 音频路径
    Frontend-->>LoadSpeech: Fbank特征 [1,T,80]
    LoadSpeech-->>Inference: batch数据
    
    Inference->>Encoder: speech [1,T,80]
    Encoder-->>Inference: encoder_out [1,T',512]
    Inference->>Adaptor: encoder_out
    Adaptor-->>Inference: adapted [1,T'',1024]
    
    Inference->>LLM: get_input_embeddings(input_ids)
    LLM-->>Inference: text_embeds [1,seq,1024]
    
    Note over Inference: 替换音频位置embedding
    
    Inference->>LLM: generate(inputs_embeds)
    LLM-->>Inference: generated_ids
    
    Inference->>Tokenizer: batch_decode
    Tokenizer-->>Inference: 识别文本
    
    Inference-->>User: 返回结果
```

## 8. 配置依赖关系图

```mermaid
flowchart TB
    subgraph Config["config.yaml 配置"]
        C1[model: FunASRNano]
        C2[audio_encoder: SenseVoiceEncoderSmall]
        C3[audio_adaptor: Transformer]
        C4[llm: Qwen3-0.6b]
        C5[frontend: WavFrontend]
    end
    
    subgraph EncoderConf["audio_encoder_conf"]
        E1[output_size: 512]
        E2[attention_heads: 4]
        E3[num_blocks: 50]
        E4[freeze: true]
    end
    
    subgraph AdaptorConf["audio_adaptor_conf"]
        A1[encoder_dim: 512]
        A2[llm_dim: 1024]
        A3[n_layer: 2]
        A4[use_low_frame_rate: true]
    end
    
    subgraph LLMConf["llm_conf"]
        L1[init_param_path: Qwen3-0.6B]
        L2[llm_dtype: bf16]
        L3[freeze: true]
        L4[use_lora: false]
    end
    
    subgraph FrontendConf["frontend_conf"]
        F1[fs: 16000]
        F2[n_mels: 80]
        F3[lfr_m: 7]
        F4[lfr_n: 6]
    end
    
    C2 --> EncoderConf
    C3 --> AdaptorConf
    C4 --> LLMConf
    C5 --> FrontendConf
    
    E1 -.->|输出维度| A1
    L2 -.->|embedding维度| A2
    
    style E1 fill:#f9f,stroke:#333
    style A1 fill:#f9f,stroke:#333
    style A2 fill:#bbf,stroke:#333
    style L2 fill:#bbf,stroke:#333
```

## 9. 特殊标记处理流程

```mermaid
flowchart TD
    subgraph Input["输入字符串"]
        I1["语音转写：&lt;|startofspeech|&gt;!/path/audio.mp3&lt;|endofspeech|&gt;"]
    end
    
    subgraph Regex["正则分割"]
        R1["pattern: &lt;|startofspeech|&gt;.*?&lt;|endofspeech|&gt;"]
        R2["splits = re.split(pattern, source_input)"]
    end
    
    subgraph Process["分片处理"]
        P1["片段1: 语音转写："]
        P2["片段2: &lt;|startofspeech|&gt;!/path/audio.mp3&lt;|endofspeech|&gt;"]
        P3["片段3: 后续文本"]
    end
    
    subgraph TextHandle["文本处理"]
        T1["tokenizer.encode()"]
        T2["fbank_mask += [0,0,...]"]
    end
    
    subgraph SpeechHandle["语音处理"]
        S1["移除标记"]
        S2["解析路径<br/>! 前缀表示文件路径<br/>!! 表示使用audio字段"]
        S3["加载音频"]
        S4["提取Fbank"]
        S5["创建fake_token占位符"]
        S6["fbank_mask += [1,1,...]"]
    end
    
    I1 --> R1 --> R2
    R2 --> P1 & P2 & P3
    
    P1 --> T1 --> T2
    P3 --> T1
    
    P2 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6
```

## 10. LoRA 微调配置图

```mermaid
flowchart TB
    subgraph LLMBase["Qwen3-0.6B Base Model"]
        B1[Frozen Weights]
        B2[28 Transformer Layers]
    end
    
    subgraph LoRAConf["LoRA 配置"]
        L1[task_type: CAUSAL_LM]
        L2[r: 16]
        L3[lora_alpha: 32]
        L4[lora_dropout: 0.05]
        L5[target_modules:<br/>q_proj, v_proj]
    end
    
    subgraph LoRALayers["LoRA 适配层"]
        LA1[q_proj LoRA]
        LA2[v_proj LoRA]
        LA3[可训练参数]
    end
    
    subgraph Training["训练配置"]
        T1[freeze_lora: true/false]
        T2[init_param_path: 可选]
    end
    
    LLMBase --> LoRALayers
    LoRAConf --> LoRALayers
    LoRALayers --> Training
    
    style B1 fill:#ccc,stroke:#333
    style LA3 fill:#9f9,stroke:#333
```

---

## 使用说明

这些 Mermaid 图可以在支持 Mermaid 的 Markdown 渲染器中查看，如：
- GitHub / GitLab
- VS Code (安装 Mermaid 插件)
- Typora
- Notion
- Obsidian

如果渲染器不支持 Mermaid，可以使用 [Mermaid Live Editor](https://mermaid.live/) 在线查看。
