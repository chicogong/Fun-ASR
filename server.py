"""Fun-ASR MLT Batch API Server

简洁的MLT多语言模型服务，支持batch批处理优化
"""

import os
import tempfile
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from contextlib import asynccontextmanager
from typing import List, Optional

from model_mlt_batch import FunASRMLT

# 配置
MODEL_PATH = os.environ.get("MODEL_PATH", "FunAudioLLM/Fun-ASR-MLT-Nano-2512")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "20"))

# 全局模型
model = None
model_kwargs = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：加载模型"""
    global model, model_kwargs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")
    print(f"📦 Loading MLT model from {MODEL_PATH}...")

    try:
        model, model_kwargs = FunASRMLT.from_pretrained(
            model=MODEL_PATH,
            device=device,
            disable_update=True
        )
        model.eval()
        print(f"✅ MLT model loaded with batch optimization")
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
):
    """单文件语音识别"""
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
            results = model.inference(
                data_in=[tmp_path],
                batch_size=1,
                language=lang,
                hotwords=hw_list if hw_list else " ",
                **model_kwargs
            )

        text = ""
        if results and len(results) > 0 and len(results[0]) > 0:
            text = results[0][0].get("text", "") if isinstance(results[0][0], dict) else ""

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
):
    """批量语音识别（batch优化）"""
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
        batch_size = min(len(tmp_paths), MAX_BATCH_SIZE)

        with torch.inference_mode():
            results = model.inference(
                data_in=tmp_paths,
                batch_size=batch_size,
                language=lang,
                hotwords=hw_list if hw_list else " ",
                **model_kwargs
            )

        # 格式化输出
        output_results = []
        if results and len(results) > 0:
            for i, result in enumerate(results[0]):
                if i < len(files):
                    text = result.get("text", "") if isinstance(result, dict) else ""
                    output_results.append({
                        "filename": files[i].filename,
                        "text": text
                    })

        return {
            "results": output_results,
            "language": lang,
            "batch_size": batch_size
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
