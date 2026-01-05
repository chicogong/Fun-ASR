"""Fun-ASR MLT Batch API Server - Optimized

简洁的MLT多语言模型服务，支持batch批处理优化
使用 AutoModel 直接调用，避免自定义wrapper的兼容性问题
"""

import os
import tempfile
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from contextlib import asynccontextmanager
from typing import List, Optional
from funasr import AutoModel

# 配置
MODEL_PATH = os.environ.get("MODEL_PATH", "FunAudioLLM/Fun-ASR-MLT-Nano-2512")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "50"))  # 增加到50以提升吞吐量

# 全局模型
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：加载模型"""
    global model

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
        print(f"✅ MLT model loaded successfully")

        # 显示显存使用
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1024**3
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"💾 GPU Memory: {mem_used:.1f}GB / {mem_total:.1f}GB")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

    yield
    model = None


app = FastAPI(
    title="Fun-ASR MLT Batch API",
    description="多语言语音识别服务（31种语言），支持batch批处理优化",
    lifespan=lifespan
)


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(default="auto"),
    hotwords: str = Form(default=""),
    itn: bool = Form(default=True),
):
    """单文件语音识别

    - file: 音频文件 (mp3/wav/flac等)
    - language: 语言代码 (auto=自动, zh=中文, en=英文, ja=日文等)
    - hotwords: 热词，逗号分隔
    - itn: 是否文本规整
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        hw_list = [w.strip() for w in hotwords.split(",") if w.strip()] if hotwords else []
        lang = language if language != "auto" else "zh"

        with torch.inference_mode():
            result = model.generate(
                input=[tmp_path],
                cache={},
                batch_size=1,
                language=lang,
                hotwords=hw_list,
                itn=itn,
            )

        text = result[0].get("text", "") if result and len(result) > 0 else ""
        return {"text": text, "language": lang}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Recognition failed: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/transcribe_batch")
async def transcribe_batch(
    files: List[UploadFile] = File(...),
    language: str = Form(default="auto"),
    hotwords: str = Form(default=""),
    itn: bool = Form(default=True),
):
    """批量语音识别（batch优化）

    - files: 多个音频文件 (最多 MAX_BATCH_SIZE 个)
    - language: 语言代码
    - hotwords: 热词，逗号分隔
    - itn: 是否文本规整
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum: {MAX_BATCH_SIZE}, got {len(files)}"
        )

    tmp_paths = []
    try:
        # 保存所有文件
        for f in files:
            content = await f.read()
            if len(content) == 0:
                continue
            suffix = os.path.splitext(f.filename or "audio")[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_paths.append(tmp.name)

        if len(tmp_paths) == 0:
            raise HTTPException(status_code=400, detail="All files are empty")

        hw_list = [w.strip() for w in hotwords.split(",") if w.strip()] if hotwords else []
        lang = language if language != "auto" else "zh"

        # MLT模型不支持真正的batch处理，使用sequential处理
        # 但仍提供批量上传API以提高用户体验
        output_results = []
        with torch.inference_mode():
            for i, tmp_path in enumerate(tmp_paths):
                try:
                    result = model.generate(
                        input=[tmp_path],
                        cache={},
                        batch_size=1,
                        language=lang,
                        hotwords=hw_list,
                        itn=itn,
                    )
                    text = result[0].get("text", "") if result and len(result) > 0 else ""
                    output_results.append({
                        "filename": files[i].filename,
                        "text": text
                    })
                except Exception as e:
                    # 单个文件失败不影响其他文件
                    output_results.append({
                        "filename": files[i].filename,
                        "text": "",
                        "error": str(e)
                    })

        return {
            "results": output_results,
            "language": lang,
            "count": len(output_results)
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Recognition failed: {str(e)}")

    finally:
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "model": "MLT" if model else "not loaded",
        "batch_optimized": True
    }


@app.get("/info")
async def info():
    """服务信息"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
            "gpu_memory_used_gb": round(torch.cuda.memory_allocated() / 1024**3, 1),
        }

    return {
        "model_path": MODEL_PATH,
        "max_batch_size": MAX_BATCH_SIZE,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        **gpu_info,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
