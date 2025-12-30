"""Fun-ASR-Nano HTTP API 服务"""

import os
import tempfile
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from funasr import AutoModel
from contextlib import asynccontextmanager

# 全局模型
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型"""
    global model
    
    model_dir = os.environ.get("MODEL_DIR", "FunAudioLLM/Fun-ASR-Nano-2512")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # model.py 位置：优先使用模型目录下的，否则用当前目录的
    model_py = os.path.join(model_dir, "model.py")
    if not os.path.exists(model_py):
        model_py = "./model.py"
    
    print(f"Loading model from {model_dir} on {device}...")
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
    语音转文字
    
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


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
