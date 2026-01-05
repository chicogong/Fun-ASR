# Fun-ASR 高精度多语言语音识别

Fun-ASR 是一个端到端的语音识别模型，支持31种语言的高精度语音转文字。

## 🚀 快速部署 MLT 多语言服务

### 一键部署脚本
```bash
# 快速部署MLT多语言模型（推荐）
./deploy.sh
```

这个脚本会自动：
- 停止现有服务
- 启动Docker容器
- 同时加载nano和mlt两个模型
- 配置环境变量
- 等待模型下载和加载
- 验证服务状态

### 服务访问
部署完成后可以访问：
- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health
- 模型列表: http://localhost:8000/models
- 语音识别: POST http://localhost:8000/asr

## 📋 支持的模型

### nano 模型 (Fun-ASR-Nano-2512)
- **优化**: 中文语音识别优化
- **语言**: 中文、英文、日文、粤语、韩文
- **特点**: 高精度中文识别，支持方言和口音

### mlt 模型 (Fun-ASR-MLT-Nano-2512)
- **优化**: 多语言支持
- **语言**: 31种语言
- **支持语言**: 中文(zh)、英文(en)、日文(ja)、韩文(ko)、粤语(yue)、越南语(vi)、印尼语(id)、泰语(th)、马来语(ms)、菲律宾语(tl)等

## 🛠️ 其他部署选项

### 手动部署方式
```bash
# 直接运行应用
python3 app.py
```

### Docker 手动部署
```bash
# 只启动nano模型
docker run -d --name funasr-nano -p 8000:8000 \
  -e LOAD_NANO=true -e LOAD_MLT=false \
  ccr.ccs.tencentyun.com/chico/funasr-server:latest

# 只启动mlt模型
docker run -d --name funasr-mlt -p 8000:8000 \
  -e LOAD_NANO=false -e LOAD_MLT=true \
  ccr.ccs.tencentyun.com/chico/funasr-server:latest

# 同时启动两个模型（推荐）
docker run -d --name funasr-dual -p 8000:8000 \
  -e LOAD_NANO=true -e LOAD_MLT=true \
  ccr.ccs.tencentyun.com/chico/funasr-server:latest
```

## 📁 项目结构

```
Fun-ASR/
├── deploy.sh              # 一键部署脚本
├── app.py                 # Gradio Web应用
├── server.py              # API服务器
├── mock_server.py         # 模拟服务器（测试用）
├── DEPLOYMENT.md          # 详细部署文档
├── Dockerfile.asr         # Docker构建文件
├── docs/                  # 文档目录
│   ├── FunASR_Architecture_Guide.md
│   ├── FunASR_Mermaid_Diagrams.md
│   └── FunASR_Quick_Reference.md
└── tests/                 # 测试目录
    └── test_*.py          # 测试脚本
```

## 🧪 测试服务

```bash
# 健康检查
curl http://localhost:8000/health

# 查看可用模型
curl http://localhost:8000/models

# 语音识别测试（需要音频文件）
curl -X POST http://localhost:8000/asr \
  -F "audio=@test.wav" \
  -F "model=mlt" \
  -F "language=zh"
```

## 📞 支持
- 查看容器日志: `docker logs -f funasr-server-dual`
- 重启服务: `docker restart funasr-server-dual`
- 停止服务: `docker stop funasr-server-dual`

更多详细信息请参考 [DEPLOYMENT.md](DEPLOYMENT.md)
