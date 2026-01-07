# Fun-ASR MLT Batch Server Docker Image
FROM ccr.ccs.tencentyun.com/chico/funasr-server:latest

WORKDIR /app

# 复制批量优化服务器和模型
COPY server_batch.py model_batch.py /app/

# 暴露端口
EXPOSE 8000

# 环境变量
ENV MODEL_PATH="FunAudioLLM/Fun-ASR-MLT-Nano-2512" \
    MAX_BATCH_SIZE="50" \
    NUM_WORKERS="4" \
    USE_GPU="true" \
    PYTHONUNBUFFERED=1

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 启动批量优化服务器
CMD ["python", "server_batch.py", "--host", "0.0.0.0", "--port", "8000"]
