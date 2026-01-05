# Fun-ASR MLT Batch Optimized Docker Image
FROM ccr.ccs.tencentyun.com/chico/funasr-server:latest

WORKDIR /app

# 复制核心文件
COPY model_mlt_batch.py /app/
COPY server.py /app/

# 暴露端口
EXPOSE 8000

# 环境变量
ENV MODEL_PATH="FunAudioLLM/Fun-ASR-MLT-Nano-2512" \
    MAX_BATCH_SIZE="20" \
    PYTHONUNBUFFERED=1

# 启动服务
CMD ["python", "server.py"]
