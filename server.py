"""Fun-ASR HTTP API 服务 (支持双模型 + Batch 推理)"""

import os
import tempfile
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from funasr import AutoModel
from contextlib import asynccontextmanager
from typing import List, Optional
import asyncio

# 全局模型
models = {}

# 最优 batch size (根据 Tesla T4 15GB 测试结果)
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "32"))

# 模型配置 (路径在运行时从环境变量读取)
MODEL_CONFIGS = {
    "nano": {
        "name": "Fun-ASR-Nano-2512",
        "languages": ["中文", "英文", "日文", "粤语", "韩文"],
        "description": "中文优化模型",
    },
    "mlt": {
        "name": "Fun-ASR-MLT-Nano-2512",
        "languages": ["zh", "en", "ja", "ko", "yue", "vi", "id", "th", "ms", "tl", "ar", "hi", "bg", "hr", "cs", "da", "nl", "et", "fi", "el", "hu", "ga", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "sv"],
        "description": "多语言模型 (31种语言)",
    },
}


def get_model_path(model_key: str) -> str:
    """获取模型路径 (运行时从环境变量读取)"""
    if model_key == "nano":
        return os.environ.get("MODEL_NANO_PATH", "FunAudioLLM/Fun-ASR-Nano-2512")
    elif model_key == "mlt":
        return os.environ.get("MODEL_MLT_PATH", "FunAudioLLM/Fun-ASR-MLT-Nano-2512")
    return ""


def get_default_model() -> str:
    """获取默认模型"""
    return os.environ.get("DEFAULT_MODEL", "nano")


def load_model(model_key: str, device: str):
    """加载指定模型"""
    config = MODEL_CONFIGS.get(model_key)
    if not config:
        raise ValueError(f"Unknown model: {model_key}")
    
    model_path = get_model_path(model_key)
    
    # 使用支持 batch 的 model_batch.py (仅 nano 模型)
    remote_code = None
    if model_key == "nano":
        model_py = "./model_batch.py"
        if os.path.exists(model_py):
            remote_code = model_py
    
    print(f"Loading {config['name']} from {model_path}...")
    if remote_code:
        print(f"  Using remote_code: {remote_code}")
    
    # 构建参数
    kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "device": device,
        "disable_update": True,
    }
    if remote_code:
        kwargs["remote_code"] = remote_code
    
    model = AutoModel(**kwargs)
    print(f"  {config['name']} loaded.")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型"""
    global models
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 加载哪些模型
    load_nano = os.environ.get("LOAD_NANO", "true").lower() == "true"
    load_mlt = os.environ.get("LOAD_MLT", "true").lower() == "true"
    
    print(f"Device: {device}")
    print(f"Load Nano: {load_nano}, Load MLT: {load_mlt}")
    print()
    
    if load_nano:
        models["nano"] = load_model("nano", device)
    
    if load_mlt:
        models["mlt"] = load_model("mlt", device)
    
    if not models:
        raise RuntimeError("No models loaded! Set LOAD_NANO=true or LOAD_MLT=true")
    
    print(f"\nAll models loaded: {list(models.keys())}")
    
    # 显存使用
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {mem_used:.1f}GB / {mem_total:.1f}GB")
    
    yield
    
    models.clear()


app = FastAPI(title="Fun-ASR API", description="支持 Nano (中文优化) 和 MLT (31种语言) 双模型", lifespan=lifespan)


def get_model(model_key: Optional[str] = None):
    """获取模型"""
    if model_key is None:
        model_key = get_default_model()
    
    if model_key not in models:
        available = list(models.keys())
        if not available:
            raise HTTPException(status_code=503, detail="No models loaded")
        # 回退到可用模型
        model_key = available[0]
    
    return models[model_key], model_key


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(default="auto"),
    model: Optional[str] = Form(default=None, description="模型: nano 或 mlt"),
    hotwords: str = Form(default=""),
    itn: bool = Form(default=True),
):
    """
    语音转文字 (单文件)
    
    - file: 音频文件 (mp3/wav/flac等)
    - language: 语言代码 (auto=自动, zh=中文, en=英文, ja=日文, ko=韩文, yue=粤语等)
    - model: 模型选择 (nano=中文优化, mlt=多语言)
    - hotwords: 热词，逗号分隔
    - itn: 是否文本规整
    """
    asr_model, model_used = get_model(model)
    
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
        hw_list = [w.strip() for w in hotwords.split(",") if w.strip()] if hotwords else []
        
        # 语言处理
        lang = language if language != "auto" else ("zh" if model_used == "mlt" else "中文")
        
        # 推理
        with torch.inference_mode():
            result = asr_model.generate(
                input=[tmp_path],
                cache={},
                batch_size=1,
                language=lang,
                hotwords=hw_list,
                itn=itn,
            )
        
        return {
            "text": result[0]["text"],
            "model": model_used,
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {str(e)}")
    
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/transcribe_batch")
async def transcribe_batch(
    files: List[UploadFile] = File(...),
    language: str = Form(default="auto"),
    model: Optional[str] = Form(default=None, description="模型: nano 或 mlt"),
    hotwords: str = Form(default=""),
    itn: bool = Form(default=True),
):
    """
    批量语音转文字 (多文件)
    
    - files: 多个音频文件 (最多 MAX_BATCH_SIZE 个)
    - language: 语言代码
    - model: 模型选择 (nano=中文优化, mlt=多语言)
    - hotwords: 热词，逗号分隔
    - itn: 是否文本规整
    """
    asr_model, model_used = get_model(model)
    
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
        
        # 解析热词
        hw_list = [w.strip() for w in hotwords.split(",") if w.strip()] if hotwords else []
        
        # 语言处理
        lang = language if language != "auto" else ("zh" if model_used == "mlt" else "中文")
        
        # Batch 推理
        batch_size = min(len(tmp_paths), MAX_BATCH_SIZE)
        with torch.inference_mode():
            results = asr_model.generate(
                input=tmp_paths,
                cache={},
                batch_size=batch_size,
                language=lang,
                hotwords=hw_list,
                itn=itn,
            )
        
        return {
            "results": [
                {"filename": files[i].filename, "text": results[i]["text"]}
                for i in range(len(results))
            ],
            "model": model_used,
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
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
    }


@app.get("/info")
async def info():
    """服务配置信息"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
            "gpu_memory_used_gb": round(torch.cuda.memory_allocated() / 1024**3, 1),
        }
    
    return {
        "models": {
            key: {
                "name": MODEL_CONFIGS[key]["name"],
                "loaded": key in models,
                "path": get_model_path(key) if key in models else None,
                "languages": MODEL_CONFIGS[key]["languages"],
                "description": MODEL_CONFIGS[key]["description"],
            }
            for key in MODEL_CONFIGS
        },
        "default_model": get_default_model(),
        "max_batch_size": MAX_BATCH_SIZE,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        **gpu_info,
    }


@app.get("/models")
async def list_models():
    """列出可用模型"""
    return {
        "available": list(models.keys()),
        "default": get_default_model(),
        "details": {
            key: {
                "name": MODEL_CONFIGS[key]["name"],
                "description": MODEL_CONFIGS[key]["description"],
                "languages": MODEL_CONFIGS[key]["languages"][:10],  # 前10种语言
            }
            for key in models
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
