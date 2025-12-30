# Fun-ASR 部署方案

## 硬件需求

| 配置 | 显存 | 推荐模型 | 吞吐量 |
|------|------|----------|--------|
| 最低 | 4GB | nano 单模型 | ~2 files/s |
| 推荐 | 8GB | nano + batch=16 | ~5 files/s |
| 高配 | 16GB | nano + mlt 双模型 | ~8 files/s |

---

## 方案一：Docker 单模型部署 (最简单)

适合：快速部署、单语言场景

```bash
# 构建镜像
docker build -f Dockerfile.gpu -t funasr-server .

# 运行 (自动下载模型)
docker run -d \
  --name funasr \
  --gpus all \
  -p 8000:8000 \
  -e LOAD_NANO=true \
  -e LOAD_MLT=false \
  funasr-server

# 或指定本地模型路径
docker run -d \
  --name funasr \
  --gpus all \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e MODEL_NANO_PATH=/models/Fun-ASR-Nano-2512 \
  -e LOAD_NANO=true \
  -e LOAD_MLT=false \
  funasr-server
```

---

## 方案二：Docker 双模型部署 (推荐)

适合：多语言场景、16GB+ 显存

```bash
docker run -d \
  --name funasr \
  --gpus all \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e MODEL_NANO_PATH=/models/Fun-ASR-Nano-2512 \
  -e MODEL_MLT_PATH=/models/Fun-ASR-MLT-Nano-2512 \
  -e LOAD_NANO=true \
  -e LOAD_MLT=true \
  -e MAX_BATCH_SIZE=32 \
  funasr-server
```

---

## 方案三：Docker Compose 部署

适合：生产环境、需要持久化配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  funasr:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    container_name: funasr-server
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - /data/models:/models:ro
      - ./logs:/app/logs
    environment:
      - MODEL_NANO_PATH=/models/Fun-ASR-Nano-2512
      - MODEL_MLT_PATH=/models/Fun-ASR-MLT-Nano-2512
      - LOAD_NANO=true
      - LOAD_MLT=true
      - MAX_BATCH_SIZE=32
      - DEFAULT_MODEL=nano
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

启动：
```bash
docker-compose up -d
```

---

## 方案四：Kubernetes 部署

适合：大规模集群、自动扩缩容

```yaml
# funasr-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: funasr
spec:
  replicas: 2
  selector:
    matchLabels:
      app: funasr
  template:
    metadata:
      labels:
        app: funasr
    spec:
      containers:
      - name: funasr
        image: your-registry/funasr-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_NANO_PATH
          value: "/models/Fun-ASR-Nano-2512"
        - name: LOAD_NANO
          value: "true"
        - name: LOAD_MLT
          value: "false"
        - name: MAX_BATCH_SIZE
          value: "32"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: funasr-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: funasr
spec:
  selector:
    app: funasr
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: funasr
spec:
  rules:
  - host: asr.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: funasr
            port:
              number: 8000
```

---

## 方案五：多 GPU 负载均衡

适合：高并发、多卡机器

```yaml
# docker-compose-multi-gpu.yml
version: '3.8'

services:
  funasr-gpu0:
    build: .
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_NANO_PATH=/models/Fun-ASR-Nano-2512
      - LOAD_NANO=true
      - LOAD_MLT=false
    volumes:
      - /data/models:/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  funasr-gpu1:
    build: .
    environment:
      - CUDA_VISIBLE_DEVICES=1
      - MODEL_NANO_PATH=/models/Fun-ASR-Nano-2512
      - LOAD_NANO=true
      - LOAD_MLT=false
    volumes:
      - /data/models:/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]

  nginx:
    image: nginx:alpine
    ports:
      - "8000:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - funasr-gpu0
      - funasr-gpu1
```

```nginx
# nginx.conf
events { worker_connections 1024; }

http {
    upstream funasr {
        least_conn;
        server funasr-gpu0:8000;
        server funasr-gpu1:8000;
    }

    server {
        listen 80;
        client_max_body_size 100M;
        
        location / {
            proxy_pass http://funasr;
            proxy_set_header Host $host;
            proxy_read_timeout 300s;
        }
    }
}
```

---

## 方案六：Conda/本地直接运行 (开发调试)

适合：开发测试、无 Docker 环境

```bash
# 创建环境
conda create -n funasr python=3.10 -y
conda activate funasr

# 安装依赖
pip install funasr transformers>=4.51.0 fastapi uvicorn python-multipart torch

# 下载模型
python -c "
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-ASR-Nano-2512', cache_dir='./models')
snapshot_download('FunAudioLLM/Fun-ASR-MLT-Nano-2512', cache_dir='./models')
"

# 启动服务
MODEL_NANO_PATH=./models/FunAudioLLM/Fun-ASR-Nano-2512 \
MODEL_MLT_PATH=./models/FunAudioLLM/Fun-ASR-MLT-Nano-2512 \
LOAD_NANO=true \
LOAD_MLT=true \
python server.py
```

---

## 环境变量说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NANO_PATH` | HuggingFace ID | Nano 模型路径 |
| `MODEL_MLT_PATH` | HuggingFace ID | MLT 模型路径 |
| `LOAD_NANO` | true | 是否加载 Nano |
| `LOAD_MLT` | true | 是否加载 MLT |
| `DEFAULT_MODEL` | nano | 默认模型 |
| `MAX_BATCH_SIZE` | 32 | 最大批处理数 |

---

## 性能调优建议

### 1. Batch Size 调优

| 显存 | 推荐 batch_size |
|------|-----------------|
| 4GB | 8 |
| 8GB | 16 |
| 16GB | 32 |
| 24GB+ | 48-64 |

### 2. 多实例 vs 大 Batch

- **低延迟场景**：多实例 + 小 batch
- **高吞吐场景**：单实例 + 大 batch

### 3. 显存不足时

```bash
# 只加载一个模型
LOAD_NANO=true
LOAD_MLT=false

# 减小 batch size
MAX_BATCH_SIZE=8
```

---

## API 调用示例

```python
import requests

# 单文件识别
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/transcribe',
        files={'file': f},
        data={'model': 'nano', 'language': '中文'}
    )
    print(response.json()['text'])

# 批量识别
files = [('files', open(f'audio{i}.wav', 'rb')) for i in range(10)]
response = requests.post(
    'http://localhost:8000/transcribe_batch',
    files=files,
    data={'model': 'nano', 'language': '中文'}
)
for result in response.json()['results']:
    print(f"{result['filename']}: {result['text']}")
```
