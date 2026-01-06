"""
Fun-ASR MLT Batch Server - 支持真正的批量并行推理
基于 model_batch.py，batch_size=6 时达到 3.5x 加速
"""

import os
import time
import tempfile
import logging
from typing import List
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# 配置
MODEL_PATH = os.getenv("MODEL_PATH", "FunAudioLLM/Fun-ASR-MLT-Nano-2512")
DEVICE = os.getenv("DEVICE", "cuda:0")
OPTIMAL_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "6"))  # 根据测试，6是最优
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "10"))

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局模型实例
model = None
model_kwargs = None

def load_model():
    """加载支持batch的MLT模型"""
    global model, model_kwargs

    logger.info(f"🔄 Loading MLT model with batch support...")
    logger.info(f"   Model: {MODEL_PATH}")
    logger.info(f"   Device: {DEVICE}")
    logger.info(f"   Optimal Batch Size: {OPTIMAL_BATCH_SIZE}")

    from model_batch import FunASRNano

    # 直接使用MODEL_PATH，让FunASRNano处理路径和配置
    model, model_kwargs = FunASRNano.from_pretrained(
        model=MODEL_PATH,
        device=DEVICE,
        disable_update=True
    )
    model.eval()

    logger.info("✅ Model loaded successfully!")

    # 预热
    logger.info("🔥 Warming up model...")
    dummy_file = "test_02.wav" if os.path.exists("test_02.wav") else None
    if dummy_file:
        try:
            _ = model.inference(data_in=[dummy_file], batch_size=1, **model_kwargs)
            logger.info("✅ Model warmed up!")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    load_model()
    yield
    # 关闭时的清理工作（如果需要）
    logger.info("👋 Shutting down...")

app = FastAPI(
    title="Fun-ASR MLT Batch Server",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """健康检查"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model": MODEL_PATH,
        "device": DEVICE,
        "optimal_batch_size": OPTIMAL_BATCH_SIZE
    }

@app.post("/transcribe_batch")
async def transcribe_batch(
    files: List[UploadFile] = File(...),
    language: str = "auto"
):
    """
    批量转录音频文件
    
    - files: 多个音频文件（建议6个为最优batch size）
    - language: 语言代码（auto/zh/en等）
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    num_files = len(files)
    
    if num_files == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if num_files > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"Too many files. Max batch size: {MAX_BATCH_SIZE}"
        )
    
    logger.info(f"📥 Received {num_files} files for batch transcription")
    
    # 保存临时文件
    tmp_dir = tempfile.mkdtemp()
    tmp_paths = []
    
    try:
        # 保存上传的文件
        for i, file in enumerate(files):
            tmp_path = os.path.join(tmp_dir, f"audio_{i}_{file.filename}")
            with open(tmp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            tmp_paths.append(tmp_path)
        
        # 批量推理
        start_time = time.time()
        batch_size = len(tmp_paths)
        
        logger.info(f"🚀 Starting batch inference (batch_size={batch_size})...")
        
        results = model.inference(
            data_in=tmp_paths,
            batch_size=batch_size,
            language=language,
            **model_kwargs
        )
        
        process_time = time.time() - start_time
        
        # 解析结果
        transcriptions = []
        for i, result in enumerate(results[0]):
            transcriptions.append({
                "filename": files[i].filename,
                "text": result.get("text", ""),
                "index": i
            })
        
        logger.info(f"✅ Batch completed in {process_time:.2f}s ({process_time/batch_size:.2f}s per file)")
        
        return JSONResponse({
            "success": True,
            "num_files": num_files,
            "batch_size": batch_size,
            "process_time": round(process_time, 2),
            "time_per_file": round(process_time / batch_size, 2),
            "results": transcriptions
        })
        
    except Exception as e:
        logger.error(f"❌ Error during batch transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 清理临时文件
        import shutil
        try:
            shutil.rmtree(tmp_dir)
        except:
            pass

@app.post("/transcribe")
async def transcribe_single(
    file: UploadFile = File(...),
    language: str = "auto"
):
    """
    单文件转录（内部调用batch接口）
    """
    return await transcribe_batch(files=[file], language=language)

@app.get("/stats")
async def get_stats():
    """获取性能统计"""
    return {
        "model": MODEL_PATH,
        "device": DEVICE,
        "optimal_batch_size": OPTIMAL_BATCH_SIZE,
        "max_batch_size": MAX_BATCH_SIZE,
        "estimated_throughput": {
            "description": "Based on batch_size=6 testing",
            "hours_per_day": 1186,
            "speedup_vs_sequential": "3.5x",
            "rtf": 0.0202
        }
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fun-ASR MLT Batch Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Fun-ASR MLT Batch Server")
    print("=" * 60)
    print(f"📍 Model: {MODEL_PATH}")
    print(f"🔧 Device: {DEVICE}")
    print(f"📊 Optimal Batch Size: {OPTIMAL_BATCH_SIZE}")
    print(f"📊 Max Batch Size: {MAX_BATCH_SIZE}")
    print(f"⚡ Expected throughput: ~1186 hours/day (single process)")
    print("=" * 60)
    print()
    
    uvicorn.run(
        "server_batch:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )
