# Conda 环境部署指南

如果你使用 Anaconda/Miniconda，推荐使用以下方式部署。

## 🚀 一键启动（推荐）

```bash
# 1. 创建并激活conda环境
conda create -n funasr python=3.8 -y
conda activate funasr

# 2. 一键安装并启动（自动安装ffmpeg和所有依赖）
chmod +x run_conda.sh
./run_conda.sh
```

脚本会自动：
- ✅ 检测conda环境
- ✅ 安装ffmpeg（如果未安装）
- ✅ 安装兼容版本的依赖
- ✅ 启动服务

## 手动安装（可选）

```bash
# 1. 激活环境
conda activate funasr

# 2. 安装系统依赖
conda install -c conda-forge ffmpeg -y

# 3. 安装Python依赖
pip install 'pydantic>=1.10.0,<2.0.0' 'fastapi>=0.95.0,<0.100.0'
pip install -r requirements.txt

# 4. 启动服务
uvicorn server_optimized:app --host 0.0.0.0 --port 8000
```

## 使用现有环境

如果你想在现有conda环境中安装：

```bash
# 1. 确保激活了正确的环境
conda activate your-env-name

# 2. 安装ffmpeg（如果未安装）
conda install ffmpeg -y

# 3. 检查并解决依赖冲突
pip install pydantic==1.10.13
pip install fastapi==0.95.2

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 启动服务
uvicorn server_optimized:app --host 0.0.0.0 --port 8000
```

## 常见问题

### 1. pydantic 版本冲突

如果遇到 `venus-boot requires pydantic<2.0` 错误：

```bash
# 降级到 pydantic 1.x
pip install 'pydantic>=1.10.0,<2.0.0' --force-reinstall
pip install 'fastapi>=0.95.0,<0.100.0' --force-reinstall
```

### 2. ffmpeg 权限错误

```bash
# 在conda环境中重新安装ffmpeg
conda install -c conda-forge ffmpeg -y
```

### 3. jupyter-server 冲突

```bash
# 升级 jupyter-server
pip install 'jupyter-server>=2.4.0,<3.0.0' --upgrade
```

## 注意事项

- ⚠️ **不要**在conda环境激活的情况下运行 `run_local.sh`
- 推荐使用独立的conda环境避免依赖冲突
- 确保 Python 版本 >= 3.8

## 验证安装

```bash
# 检查ffmpeg
ffmpeg -version

# 检查Python包
python -c "from fastapi import FastAPI; from funasr import AutoModel; print('✅ 安装成功')"

# 健康检查
curl http://localhost:8000/health
```
