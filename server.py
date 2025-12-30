"""Fun-ASR-Nano HTTP API 服务 (支持 Batch 推理)"""

import os
import tempfile
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from funasr import AutoModel
from contextlib import asynccontextmanager
from typing import List
import asyncio

# 全局模型
model = None

# 最优 batch size (根据 Tesla T4 15GB 测试结果)
# batch=30-40 时吞吐量最高 (~8.5 files/sec)
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "32"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型"""
    global model
    
    model_dir = os.environ.get("MODEL_PATH", os.environ.get("MODEL_DIR", "FunAudioLLM/Fun-ASR-Nano-2512"))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 使用支持 batch 的 model_batch.py
    model_py = "./model_batch.py"
    if not os.path.exists(model_py):
        model_py = os.path.join(model_dir, "model.py")
    
    print(f"Loading model from {model_dir} on {device}...")
    print(f"Using remote_code: {model_py}")
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code=model_py,
        device=device,
        disable_update=True,
    )
    print("Model loaded.")
    
    yield
    
    model = None


app = FastAPI(title="Fun-ASR-Nano API", lifespan=lifespan)


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(default="中文"),
    hotwords: str = Form(default=""),
    itn: bool = Form(default=True),
):
    """
    语音转文字 (单文件)
    
    - file: 音频文件 (mp3/wav/flac等)
    - language: 语言 (中文/英文/日文/粤语等)
    - hotwords: 热词，逗号分隔
    - itn: 是否文本规整
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 检查文件
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    
    # 保存临时文件
    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # 解析热词
        hw_list = [w.strip() for w in hotwords.split(",") if w.strip()]
        
        # 推理
        with torch.inference_mode():
            result = model.generate(
                input=[tmp_path],
                cache={},
                batch_size=1,
                language=language,
                hotwords=hw_list,
                itn=itn,
            )
        
        return {"text": result[0]["text"]}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {str(e)}")
    
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/transcribe_batch")
async def transcribe_batch(
    files: List[UploadFile] = File(...),
    language: str = Form(default="中文"),
    hotwords: str = Form(default=""),
    itn: bool = Form(default=True),
):
    """
    批量语音转文字 (多文件)
    
    - files: 多个音频文件 (最多 MAX_BATCH_SIZE 个)
    - language: 语言 (中文/英文/日文/粤语等)
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
            detail=f"Too many files. Maximum batch size is {MAX_BATCH_SIZE}, got {len(files)}"
        )
    
    # 保存所有临时文件
    tmp_paths = []
    try:
        for file in files:
            content = await file.read()
            if len(content) == 0:
                continue
            suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_paths.append(tmp.name)
        
        if len(tmp_paths) == 0:
            raise HTTPException(status_code=400, detail="All files are empty")
        
        # 解析热词
        hw_list = [w.strip() for w in hotwords.split(",") if w.strip()]
        
        # Batch 推理 (使用最优 batch size)
        batch_size = min(len(tmp_paths), MAX_BATCH_SIZE)
        with torch.inference_mode():
            results = model.generate(
                input=tmp_paths,
                cache={},
                batch_size=batch_size,
                language=language,
                hotwords=hw_list,
                itn=itn,
            )
        
        return {
            "results": [
                {"filename": files[i].filename, "text": results[i]["text"]}
                for i in range(len(results))
            ],
            "batch_size": batch_size,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {str(e)}")
    
    finally:
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/info")
async def info():
    """服务配置信息"""
    return {
        "max_batch_size": MAX_BATCH_SIZE,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if torch.cuda.is_available() else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
