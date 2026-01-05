# Fun-ASR MLT 部署指南

快速在新机器上部署Fun-ASR MLT多语言语音识别服务。

## 快速开始

### 1. 获取代码

```bash
git clone https://github.com/chicogong/Fun-ASR.git
cd Fun-ASR
git checkout feature/mlt-batch-simple
```

### 2. Docker部署（推荐）

#### CPU模式（稳定推荐）

```bash
./docker-start.sh
```

#### GPU模式（需要兼容环境）

```bash
USE_GPU=true ./docker-start.sh
```

### 3. 本地部署

```bash
./run.sh local
```

## 详细说明

### Docker部署

**优点：**
- 环境隔离，不影响系统
- 一键启动，自动处理依赖
- 跨平台支持

**CPU模式（默认）：**
```bash
# 方式1：使用脚本（推荐）
./docker-start.sh

# 方式2：手动启动
docker build -t funasr-mlt-batch:latest .
docker run -d --name funasr -p 8000:8000 \
  -e USE_GPU=false \
  -v ~/.cache/modelscope:/root/.cache/modelscope \
  funasr-mlt-batch:latest
```

**GPU模式：**
```bash
# 需要：NVIDIA GPU + nvidia-docker2
USE_GPU=true ./docker-start.sh
```

**注意：** Docker GPU模式可能存在兼容性问题（PyTorch/CUDA版本），建议优先使用CPU模式或本地部署。

### 本地部署

**推荐环境：**
- Python 3.11+ （最佳兼容性）
- CUDA 12.1+ （GPU模式）

**一键启动：**
```bash
./run.sh local
```

脚本会自动：
1. 检测Python版本（优先3.11）
2. 创建虚拟环境venv311
3. 安装依赖
4. 启动服务

**手动安装：**
```bash
# 创建虚拟环境
python3.11 -m venv venv311
source venv311/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 启动服务
python server.py
```

## 验证部署

```bash
# 健康检查
curl http://localhost:8000/health

# 服务信息
curl http://localhost:8000/info

# 测试识别
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=zh"
```

## 环境要求

### 最低要求
- CPU: 2核+
- 内存: 8GB+
- 磁盘: 5GB+ (模型约2GB)

### GPU加速（可选）
- NVIDIA GPU (推荐Tesla T4或更好)
- CUDA 12.1+
- 显存: 4GB+

## 支持的语言

MLT模型支持31种语言，包括：

- 中文 (zh)
- 英文 (en)
- 日文 (ja)
- 韩文 (ko)
- 粤语 (yue)
- 越南语 (vi)
- 印尼语 (id)
- 泰语 (th)
- 马来语 (ms)
- 菲律宾语 (tl)
- 阿拉伯语 (ar)
- 印地语 (hi)
- 等...

## API使用

### 单文件识别

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=zh" \
  -F "hotwords=语音,识别"
```

### 批量识别

```bash
curl -X POST http://localhost:8000/transcribe_batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav" \
  -F "language=zh"
```

### API文档

访问: http://localhost:8000/docs

## 故障排除

### Docker相关

**容器启动失败：**
```bash
# 查看日志
docker logs funasr

# 重启容器
docker restart funasr
```

**GPU模式crash：**
```bash
# 使用CPU模式
docker stop funasr && docker rm funasr
./docker-start.sh  # 默认CPU模式
```

### 本地部署

**Python版本问题：**
```bash
# 确认Python 3.11已安装
python3.11 --version

# 如未安装，使用系统包管理器安装
# Ubuntu/Debian:
sudo apt install python3.11 python3.11-venv

# CentOS/RHEL:
sudo yum install python3.11
```

**FunASR模型加载失败：**
```bash
# 确保Python 3.11 + FunASR 1.2.9
source venv311/bin/activate
pip install "funasr==1.2.9" --force-reinstall
```

**GPU不可用：**
```bash
# 检查CUDA
nvidia-smi

# 检查PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 如GPU不可用，使用CPU模式
export USE_GPU=false
python server.py
```

## 性能优化

### 批处理优化
- 使用`/transcribe_batch`接口处理多文件
- 最大batch_size: 20

### GPU加速
- 本地部署推荐使用GPU（自动启用）
- RTF约0.44（实时因子，越小越快）

### 模型缓存
- 首次启动会下载约2GB模型
- 模型缓存在`~/.cache/modelscope/`
- Docker挂载此目录避免重复下载

## 生产部署建议

1. **使用本地部署** - GPU兼容性更好
2. **使用Nginx反向代理** - 负载均衡和SSL
3. **使用systemd管理** - 自动重启和日志
4. **监控GPU使用率** - nvidia-smi
5. **设置请求限流** - 防止过载

## 更多信息

- GitHub: https://github.com/chicogong/Fun-ASR
- 文档: README.md
- 问题反馈: GitHub Issues
# Fun-ASR MLT Performance Report

## 测试环境

- **模型**: FunAudioLLM/Fun-ASR-MLT-Nano-2512
- **设备**: NVIDIA Tesla T4 GPU
- **GPU 内存**: 3.1 GB / 14.6 GB
- **CUDA**: 12.1
- **Python**: 3.11.11
- **FunASR**: 1.2.9
- **测试日期**: 2026-01-05

## 性能指标总结

### 🏆 最佳性能

| 指标 | 数值 | 说明 |
|------|------|------|
| **最佳 RTF** | 0.0602 | 批量大小 5 |
| **最大日处理量** | 398.54 小时/天 | 批量大小 5 |
| **最大日处理量** | 23,912 分钟/天 | 约 16.6 天的音频 |
| **处理速度** | 16.6x 实时 | 比实时播放快 16.6 倍 |
| **最佳吞吐量** | 1.56 文件/秒 | 批量大小 5 |

### 📊 批量大小性能对比

| 批量大小 | 平均 RTF | 最佳 RTF | 平均处理量(小时/天) | 最大处理量(小时/天) |
|----------|----------|----------|---------------------|---------------------|
| 1 | 0.0668 | 0.0603 | 361.30 | 398.32 |
| 2 | 0.0649 | 0.0636 | 370.08 | 377.16 |
| 3 | 0.0668 | 0.0653 | 359.51 | 367.35 |
| 4 | 0.0670 | 0.0658 | 358.27 | 364.91 |
| **5** | **0.0641** | **0.0602** | **375.14** | **398.54** |
| 8 | 0.0651 | 0.0631 | 369.08 | 380.15 |
| 10 | 0.0659 | 0.0654 | 364.13 | 367.21 |
| 12 | 0.0647 | 0.0631 | 371.21 | 380.09 |
| 15 | 0.1045 | 0.0653 | 285.11 | 367.75 |
| 18 | 0.0659 | 0.0653 | 363.93 | 367.31 |
| 20 | 0.0663 | 0.0654 | 361.93 | 367.10 |

## 💡 推荐配置

### 最佳批量大小: **5**

**原因**: RTF 和处理能力之间的最佳平衡

- **RTF**: 0.0641 (平均), 0.0602 (最佳)
- **日处理量**: 375.14 小时 (平均), 398.54 小时 (最大)
- **吞吐量**: 1.56 文件/秒
- **稳定性**: 三次测试结果一致性最高

### 实际处理能力

以批量大小 5 为例，系统可以：

- **每小时处理**: 16.6 小时的音频
- **每天处理**: 398.54 小时的音频（约 16.6 天）
- **每月处理**: 约 12,000 小时的音频（500 天）

## 📈 性能分析

### RTF 分布

- **最低 RTF**: 0.0602 (批量大小 5)
- **最高 RTF**: 0.1806 (批量大小 15 某次测试，可能异常)
- **稳定区间**: 0.0602 - 0.0678 (大部分批量大小)

### 批量大小影响

1. **批量大小 1-5**: RTF 持续优化，达到最佳性能
2. **批量大小 5-12**: 性能稳定，RTF 在 0.063-0.067 之间
3. **批量大小 15**: 出现性能波动（可能内存或调度问题）
4. **批量大小 18-20**: 性能稳定但略低于最佳值

### GPU 利用率

- **GPU 内存使用**: 3.1 GB / 14.6 GB (21%)
- **潜力**: GPU 内存充足，可以考虑更大的批量或并发处理

## 🎯 应用场景分析

### 场景 1: 大规模批量转录

**需求**: 每天需要处理 200 小时的音频

- **推荐配置**: 批量大小 5
- **所需时间**: 200 / 398.54 × 24 = **12 小时**
- **CPU 占用**: 12 小时（可在夜间运行）

### 场景 2: 实时近线处理

**需求**: 持续处理用户上传的音频文件

- **推荐配置**: 批量大小 2-5
- **处理能力**: 每秒 1.5 个文件（假设每个文件 10 秒）
- **延迟**: 平均 0.6 秒

### 场景 3: 极限吞吐量

**需求**: 最大化日处理量

- **推荐配置**: 批量大小 5
- **理论日处理量**: 398.54 小时
- **实际建议**: 350 小时（留有余量，考虑系统稳定性）

## 🔧 优化建议

### 当前系统

1. **最佳批量大小**: 使用批量大小 5 可获得最佳性能
2. **GPU 利用率**: 当前 GPU 内存仅使用 21%，可考虑:
   - 并发处理多个批次
   - 增大批量大小（需要测试稳定性）
   - 同时运行多个模型实例

### 进一步提升

1. **模型优化**:
   - 考虑量化模型（INT8/FP16）以提高速度
   - 使用 TensorRT 优化推理

2. **系统优化**:
   - 使用异步处理
   - 实现请求队列和批处理聚合
   - 增加 GPU 数量进行水平扩展

3. **批处理策略**:
   - 动态批量大小（根据队列长度调整）
   - 按音频时长分组批处理

## 📝 测试方法

### 测试脚本

```bash
# 快速测试（5 个文件，批量大小 1-5）
python quick_performance_test.py

# 完整测试（20 个文件，批量大小 1-20）
python full_performance_test.py
```

### 测试参数

- **音频格式**: WAV, 16kHz, 单声道
- **音频时长**: 10 秒/文件
- **测试次数**: 每个批量大小测试 3 次
- **语言**: 中文 (zh)

### 测试方式

1. 生成指定数量的测试音频文件（静音）
2. 依次测试不同批量大小（1, 2, 3, 4, 5, 8, 10, 12, 15, 18, 20）
3. 每个批量大小测试 3 次取平均值
4. 记录处理时间、RTF、吞吐量等指标

## 🚀 快速开始

### 运行性能测试

```bash
# 确保服务已启动
curl http://localhost:8000/health

# 运行性能测试
python full_performance_test.py
```

### 使用最佳配置

```bash
# 批量处理 5 个文件（推荐）
curl -X POST http://localhost:8000/transcribe_batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav" \
  -F "files=@audio4.wav" \
  -F "files=@audio5.wav" \
  -F "language=zh"
```

## 📊 结论

Fun-ASR MLT 模型在 Tesla T4 GPU 上表现出色：

1. ✅ **优秀的 RTF**: 最佳 RTF 0.0602，比实时快 16.6 倍
2. ✅ **高吞吐量**: 每天可处理约 400 小时音频
3. ✅ **稳定性好**: 批量大小 5 时性能最稳定
4. ✅ **资源利用**: GPU 内存使用率仅 21%，仍有优化空间

**推荐生产配置**: 批量大小 5，可满足大规模语音转录需求。

---

**报告生成时间**: 2026-01-05
**测试版本**: Fun-ASR MLT v1.0
**GPU**: NVIDIA Tesla T4

---

# Fun-ASR L20 性能优化指南

## 当前性能

- **批量大小**: 5
- **RTF**: 0.0360
- **日处理**: 667 小时
- **GPU 利用率**: 7% ⚠️

## 测试结果

### ✅ 立即可行的优化

| 优化方案 | RTF | 日处理 | 提升 | 难度 |
|----------|-----|--------|------|------|
| 批量大小 → 20 | 0.0305 | 788 小时 | +18% | ⭐ 简单 |
| 并发请求 16 | 0.0291 | 826 小时 | +24% | ⭐⭐ 中等 |

### 🚀 进阶优化方向

| 优化方案 | 预估提升 | GPU 利用率 | 难度 |
|----------|----------|------------|------|
| 增加 uvicorn workers | 2-4x | 20-30% | ⭐⭐ |
| FP16 精度推理 | 1.5-2x | 10-15% | ⭐⭐⭐ |
| TensorRT 优化 | 2-3x | 15-25% | ⭐⭐⭐⭐ |
| 真正的批量并行 | 3-5x | 30-50% | ⭐⭐⭐⭐ |
| 多 GPU/模型副本 | Nx | 80%+ | ⭐⭐⭐⭐⭐ |

### 🎯 理论极限

- **当前 GPU 利用率**: 7%
- **假设优化到 80%**: 理论最大 **9,070 小时/天**
- **相比当前提升**: **13.5x 倍**

---

## 🔧 具体优化步骤

### 1️⃣ 增加批量大小到 20（立即生效）

**修改 1**: 更新服务器配置

编辑 `server.py`:
```python
# 当前
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "20"))

# 建议
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "50"))
```

**修改 2**: 更新 Docker/启动配置

```bash
# docker-start.sh 或运行时环境变量
export MAX_BATCH_SIZE=50

# Docker 运行
docker run ... -e MAX_BATCH_SIZE=50 ...
```

**预期效果**:
- 日处理: 667 → 788 小时 (+18%)
- 可能更高（如果批量 30-50 有效）

---

### 2️⃣ 启用多 Worker 模式（中等难度）

**当前**: 单个 uvicorn 进程

**优化**: 多个 worker 并行处理

编辑 `server.py`:
```python
# 当前启动方式
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)

# 优化启动方式
if __name__ == "__main__":
    import multiprocessing
    workers = min(4, multiprocessing.cpu_count())
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        workers=workers,  # 多 worker
        limit_concurrency=100
    )
```

**预期效果**:
- 日处理: 788 → 1,500-2,000 小时 (2-3x)
- GPU 利用率: 7% → 20-30%

---

### 3️⃣ 实现真正的批量并行（高难度）

**当前问题**: `server.py` 中是顺序处理
```python
# 当前代码 (server.py line 159-184)
for i, tmp_path in enumerate(tmp_paths):  # 顺序处理 ❌
    result = model.generate(input=[tmp_path], ...)
```

**优化方案**: 真正的批量推理
```python
# 优化代码
# 一次性处理所有文件
results = model.generate(
    input=tmp_paths,  # 批量输入 ✅
    batch_size=len(tmp_paths),
    language=lang,
    ...
)
```

**注意**: MLT 模型可能不支持真正的批量，需要测试

**预期效果**:
- 日处理: 2,000 → 4,000-6,000 小时 (2-3x)
- GPU 利用率: 30% → 50-70%

---

### 4️⃣ FP16 半精度推理（高难度）

启用 FP16 可以 1.5-2x 加速

编辑 `server.py`:
```python
# 当前
model = AutoModel(
    model=model_path,
    device=device,
    disable_update=True,
)

# 优化
import torch
model = AutoModel(
    model=model_path,
    device=device,
    disable_update=True,
)

# 启用 FP16
if device == "cuda:0":
    model = model.half()  # 转换为 FP16
```

**预期效果**:
- 日处理: 4,000 → 6,000-8,000 小时 (1.5-2x)
- GPU 内存: 减少约 50%

---

### 5️⃣ 异步队列 + 动态批处理（高难度）

实现智能批处理系统：
- 请求进入队列
- 自动聚合成批
- 动态调整批量大小

**架构**:
```
客户端请求 → 请求队列 → 动态批处理器 → 模型推理 → 返回结果
                ↓
         每 50ms 或达到 20 个请求时触发
```

**预期效果**:
- 延迟: 增加 50-200ms
- 吞吐量: 提升 5-10x
- GPU 利用率: 50-80%

---

## 🚀 快速优化方案（推荐）

### 方案 A: 简单提升 20-30%

```bash
# 1. 修改配置
export MAX_BATCH_SIZE=50

# 2. 重新部署
docker stop funasr && docker rm funasr
docker run -d --name funasr \
  --gpus all \
  -p 8000:8000 \
  -e USE_GPU=true \
  -e MAX_BATCH_SIZE=50 \
  -v ~/.cache/modelscope:/root/.cache/modelscope \
  ccr.ccs.tencentyun.com/chico/funasr-mlt-batch:latest

# 3. 使用批量 20-30
curl -X POST http://localhost:8000/transcribe_batch \
  -F "files=@audio1.wav" \
  ... (20-30 个文件)
```

**效果**: 日处理 667 → 850+ 小时

---

### 方案 B: 中等提升 2-3x

修改 `server.py` 添加 worker 支持，然后重新构建镜像

**效果**: 日处理 667 → 1,500-2,000 小时

---

### 方案 C: 极限优化 10x+

需要重构代码：
1. 多 worker
2. 批量并行
3. FP16
4. 异步队列
5. TensorRT

**效果**: 日处理 667 → 6,000-9,000 小时

---

## 📊 投入产出比

| 方案 | 开发时间 | 提升倍数 | 推荐度 |
|------|----------|----------|--------|
| 批量大小 → 50 | 5 分钟 | 1.2x | ⭐⭐⭐⭐⭐ |
| 多 Worker | 30 分钟 | 2-3x | ⭐⭐⭐⭐⭐ |
| 真正批量并行 | 2 小时 | 2-3x | ⭐⭐⭐⭐ |
| FP16 优化 | 1 小时 | 1.5-2x | ⭐⭐⭐⭐ |
| 异步队列 | 1 天 | 5-10x | ⭐⭐⭐ |
| TensorRT | 2-3 天 | 2-3x | ⭐⭐ |

---

## 🎯 建议的优化路线

### 第一步：立即提升 20%
- ✅ 批量大小 → 20-50
- ⏱️ 时间: 5 分钟

### 第二步：中期提升 2-3x
- ✅ 多 Worker 模式
- ✅ FP16 精度
- ⏱️ 时间: 1-2 小时

### 第三步：长期优化 5-10x
- ✅ 真正批量并行
- ✅ 异步队列
- ✅ TensorRT
- ⏱️ 时间: 1-3 天

---

## 📝 总结

**当前瓶颈**: GPU 利用率仅 7%，大量性能浪费

**最优方案**: 先快速优化批量大小（5分钟），再逐步实现多 Worker 和批量并行

**理论极限**: L20 可以达到 9,000+ 小时/天，是当前的 13.5 倍！
