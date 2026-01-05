# Fun-ASR Batch处理优化方案

## 概述

Fun-ASR的batch处理优化显著提升了多文件处理性能，尤其是MLT和Nano模型。

## 性能对比

### Nano模型
- **标准版本**: RTF ~0.375 (batch_size=1)
- **优化版本**: RTF ~0.112 (batch_size=6)
- **性能提升**: 70%+

### MLT模型
- **标准版本**: 不支持batch处理
- **优化版本**: RTF ~0.054 (batch_size=4)
- **性能提升**: 支持batch + 80%性能提升

## 日处理能力

### Nano模型（优化后）
- **batch_size=6**: RTF=0.112
- **24小时运行**: ~214小时音频/天
- **8小时运行**: ~71小时音频/天

### MLT模型（优化后）
- **batch_size=4**: RTF=0.054
- **24小时运行**: ~444小时音频/天
- **8小时运行**: ~148小时音频/天

## 方案1：本地部署（推荐用于开发测试）

### 环境要求
```bash
Python >= 3.11
PyTorch >= 2.0
FunASR >= 1.2
```

### 安装依赖
```bash
pip install funasr modelscope soundfile torch transformers
```

### 运行测试
```bash
# Nano模型测试
python3.11 test_local_batch.py

# MLT模型测试
python3.11 test_mlt_batch.py
```

### 启动本地服务
```bash
python3.11 server_optimized.py
```

### 优点
- 快速测试和开发
- 完全控制环境
- 易于调试

### 缺点
- 需要手动配置环境
- 依赖管理复杂

## 方案2：Docker镜像部署（推荐用于生产）

### 构建优化镜像

#### 方法A：自动构建
```bash
./build_optimized_docker.sh
```

#### 方法B：手动构建
```bash
docker build -t funasr-batch-optimized:latest -f Dockerfile.optimized .
```

### 启动容器
```bash
docker run -d \\
  -p 8000:8000 \\
  --name funasr-opt \\
  funasr-batch-optimized:latest
```

### 查看日志
```bash
docker logs -f funasr-opt
```

### 测试服务
```bash
# 健康检查
curl http://localhost:8000/health

# 查看模型列表
curl http://localhost:8000/models

# 单文件测试
curl -X POST http://localhost:8000/transcribe \\
  -F "file=@test.wav" \\
  -F "model=nano"

# Batch测试
curl -X POST http://localhost:8000/transcribe_batch \\
  -F "files=@test1.wav" \\
  -F "files=@test2.wav" \\
  -F "files=@test3.wav" \\
  -F "model=mlt"
```

### 优点
- 环境一致性
- 易于部署和扩展
- 生产就绪

### 缺点
- 首次构建需要时间
- 镜像体积较大

## API使用示例

### Python客户端
```python
import requests

# 单文件转写
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/transcribe',
        files={'file': f},
        data={'model': 'nano', 'language': 'zh'}
    )
    print(response.json()['text'])

# Batch转写
files = [
    ('files', open('audio1.wav', 'rb')),
    ('files', open('audio2.wav', 'rb')),
    ('files', open('audio3.wav', 'rb')),
]
response = requests.post(
    'http://localhost:8000/transcribe_batch',
    files=files,
    data={'model': 'mlt', 'language': 'auto'}
)
for result in response.json()['results']:
    print(f"{result['filename']}: {result['text']}")
```

### cURL示例
```bash
# 单文件
curl -X POST http://localhost:8000/transcribe \\
  -F "file=@audio.wav" \\
  -F "model=nano" \\
  -F "language=zh"

# Batch处理
curl -X POST http://localhost:8000/transcribe_batch \\
  -F "files=@audio1.wav" \\
  -F "files=@audio2.wav" \\
  -F "model=mlt"
```

## 核心优化技术

### 1. Left Padding优化 (Nano模型)
```python
# 关键代码在 model_batch.py
def inference(self, data_in, batch_size=1, **kwargs):
    # Left padding确保batch内序列对齐
    max_len = max([x.size(0) for x in input_ids])
    padded_inputs = [
        torch.cat([torch.zeros(max_len - x.size(0)), x])
        for x in input_ids
    ]
```

### 2. Batch Wrapper (MLT模型)
```python
# model_mlt_batch.py
class MLTBatchWrapper:
    def generate(self, input, **kwargs):
        # 直接支持batch处理
        return self.model.generate(
            input=input,
            batch_size=kwargs.get('batch_size', len(input))
        )
```

## 故障排除

### 问题1：模型加载失败
```
AssertionError: FunASRNano is not registered
```
**解决方案**: 确保使用优化版server (server_optimized.py)，它直接导入batch优化模型类

### 问题2：Batch处理失败
```
batch decoding is not implemented
```
**解决方案**: 使用方案2（Docker镜像），确保使用batch优化版本

### 问题3：内存不足
**解决方案**: 降低batch_size或使用CPU模式

## 性能调优建议

1. **Batch Size选择**:
   - Nano: 推荐 batch_size=6
   - MLT: 推荐 batch_size=4
   - 根据内存调整

2. **并发处理**:
   - 单个请求使用batch
   - 多个请求可以并发

3. **硬件配置**:
   - CPU: 推荐16核心以上
   - GPU: 推荐Tesla T4或更好
   - 内存: 推荐32GB+

## 下一步

- [x] 本地测试验证
- [x] Docker镜像构建
- [ ] 生产环境部署
- [ ] 监控和日志
- [ ] 性能基准测试

## 参考

- [FunASR官方文档](https://github.com/alibaba-damo-academy/FunASR)
- [ModelScope](https://modelscope.cn/)
