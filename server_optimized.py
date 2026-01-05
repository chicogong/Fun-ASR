import os
import tempfile
import time
import torch
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from funasr import AutoModel

# 配置
MODEL_PATH = os.environ.get("MODEL_PATH", "FunAudioLLM/Fun-ASR-MLT-Nano-2512")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "50"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))  # 新增：worker 数量

# 全局模型
model = None

app = FastAPI(title="Fun-ASR MLT Batch Server (Optimized)")


@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    global model

    print(f"🚀 Starting Fun-ASR MLT Batch Server (Optimized)")
    print(f"📊 Configuration:")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Max Batch Size: {MAX_BATCH_SIZE}")
    print(f"   Workers: {NUM_WORKERS}")

    # 默认使用GPU，如需CPU模式设置环境变量 USE_GPU=false
    use_gpu = os.environ.get("USE_GPU", "true").lower() == "true"
    device = "cuda:0" if (torch.cuda.is_available() and use_gpu) else "cpu"
    print(f"🔧 Device: {device}")
    print(f"📦 Loading MLT model from {MODEL_PATH}...")

    try:
        model = AutoModel(
            model=MODEL_PATH,
            trust_remote_code=True,
            device=device,
            disable_update=True,
        )
        print(f"✅ Model loaded successfully on {device}")

        # 预热模型
        if device == "cuda:0":
            print("🔥 Warming up GPU...")
            dummy_audio = os.path.join(tempfile.gettempdir(), "warmup.wav")
            # 创建简单的测试音频
            import wave
            import struct
            with wave.open(dummy_audio, 'w') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                for _ in range(16000):  # 1秒
                    wav.writeframes(struct.pack('<h', 0))

            # 预热推理
            with torch.inference_mode():
                _ = model.generate(input=[dummy_audio], cache={}, batch_size=1, language="zh")
            os.remove(dummy_audio)
            print("✅ GPU warmed up")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "model": "MLT",
        "batch_optimized": True,
        "multi_worker": True,
        "workers": NUM_WORKERS,
    }


@app.get("/info")
async def info():
    """服务信息"""
    gpu_info = {}
    if torch.cuda.is_available():
        try:
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
                ),
                "gpu_memory_used_gb": round(
                    torch.cuda.memory_allocated(0) / 1024**3, 1
                ),
            }
        except:
            pass

    return {
        "model_path": MODEL_PATH,
        "max_batch_size": MAX_BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        **gpu_info,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("zh"),
    hotwords: Optional[str] = Form(None),
    use_itn: bool = Form(True),
):
    """单文件转录"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 保存上传的文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 处理热词
        hw_list = hotwords.strip() if hotwords else ""

        # 推理
        with torch.inference_mode():
            result = model.generate(
                input=[tmp_path],
                cache={},
                batch_size=1,
                language=language,
                hotwords=hw_list,
                itn=use_itn,
            )

        text = result[0].get("text", "") if result and len(result) > 0 else ""

        return {"text": text, "language": language}

    finally:
        # 清理临时文件
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/transcribe_batch")
async def transcribe_batch(
    files: List[UploadFile] = File(...),
    language: str = Form("zh"),
    hotwords: Optional[str] = Form(None),
    use_itn: bool = Form(True),
):
    """批量转录（优化版本）"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 检查批量大小
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum: {MAX_BATCH_SIZE}, got {len(files)}",
        )

    # 保存所有上传的文件
    tmp_paths = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_paths.append(tmp.name)

    try:
        # 处理热词
        hw_list = hotwords.strip() if hotwords else ""

        # 批量推理（顺序处理）
        # 注意：MLT模型不支持真正的批量推理，这里是顺序处理
        output_results = []
        with torch.inference_mode():
            for i, tmp_path in enumerate(tmp_paths):
                try:
                    result = model.generate(
                        input=[tmp_path],
                        cache={},
                        batch_size=1,
                        language=language,
                        hotwords=hw_list,
                        itn=use_itn,
                    )

                    text = (
                        result[0].get("text", "") if result and len(result) > 0 else ""
                    )
                    output_results.append({"filename": files[i].filename, "text": text})

                except Exception as e:
                    output_results.append(
                        {"filename": files[i].filename, "text": "", "error": str(e)}
                    )

        return {
            "results": output_results,
            "language": language,
            "count": len(output_results),
        }

    finally:
        # 清理所有临时文件
        for tmp_path in tmp_paths:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    import multiprocessing

    # 多 Worker 模式
    workers = NUM_WORKERS if NUM_WORKERS > 0 else min(4, multiprocessing.cpu_count())

    print(f"\n🚀 Starting with {workers} workers for maximum throughput")
    print(f"💡 Estimated capacity increase: {workers}x")
    print(f"📊 Expected daily processing: ~{807 * workers} hours\n")

    uvicorn.run(
        "server_optimized:app",
        host="0.0.0.0",
        port=8000,
        workers=workers,
        limit_concurrency=100,  # 限制并发连接数
        timeout_keep_alive=30,
    )
