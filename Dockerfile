# Fun-ASR MLT Batch Server Docker Image
FROM ccr.ccs.tencentyun.com/chico/funasr-server:latest

WORKDIR /app

# 复制优化版服务器（多 Worker 支持）
COPY server_optimized.py /app/

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

# 启动服务（多 Worker 模式）
CMD ["python", "server_optimized.py"]
