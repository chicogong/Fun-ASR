# Fun-ASR MLT Batch Server Docker Image
FROM ccr.ccs.tencentyun.com/chico/funasr-server:latest

WORKDIR /app

# 只复制服务器文件（使用AutoModel，无需自定义wrapper）
COPY server.py /app/

# 暴露端口
EXPOSE 8000

# 环境变量
ENV MODEL_PATH="FunAudioLLM/Fun-ASR-MLT-Nano-2512" \
    MAX_BATCH_SIZE="20" \
    USE_GPU="true" \
    PYTHONUNBUFFERED=1

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 启动服务
CMD ["python", "server.py"]
