"""Fun-ASR HTTP API 服务 - Batch优化版本

此版本直接集成batch优化，绕过AutoModel的限制
"""

import os
import tempfile
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from contextlib import asynccontextmanager
from typing import List, Optional

# 导入batch优化模型
from model_batch import FunASRNano
from model_mlt_batch import FunASRMLT

# 全局模型
models = {}

# 配置
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "20"))

MODEL_CONFIGS = {
    "nano": {
        "name": "Fun-ASR-Nano-2512",
        "path": os.environ.get("MODEL_NANO_PATH", "FunAudioLLM/Fun-ASR-Nano-2512"),
        "description": "中文优化模型 (Batch优化)",
    },
    "mlt": {
        "name": "Fun-ASR-MLT-Nano-2512",
        "path": os.environ.get("MODEL_MLT_PATH", "FunAudioLLM/Fun-ASR-MLT-Nano-2512"),
        "description": "多语言模型 (31种语言, Batch优化)",
    },
}


def load_model(model_key: str, device: str):
    """加载batch优化模型"""
    config = MODEL_CONFIGS.get(model_key)
    if not config:
        raise ValueError(f"Unknown model: {model_key}")

    model_path = config["path"]

    print(f"Loading {config['name']} (Batch-Optimized) from {model_path}...")

    try:
        if model_key == "nano":
            model, kwargs = FunASRNano.from_pretrained(
                model=model_path,
                device=device,
                disable_update=True,
                remote_code="./model_batch.py"
            )
        elif model_key == "mlt":
            model, kwargs = FunASRMLT.from_pretrained(
                model=model_path,
                device=device,
                disable_update=True
            )
        else:
            raise ValueError(f"Unknown model key: {model_key}")

        model.eval()
        print(f"  ✅ {config['name']} loaded with batch optimization")
        return model, kwargs

    except Exception as e:
        print(f"  ❌ Failed to load {model_key}: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")

    # 加载所有模型
    for model_key in MODEL_CONFIGS.keys():
        try:
            model, kwargs = load_model(model_key, device)
            models[model_key] = {"model": model, "kwargs": kwargs}
        except Exception as e:
            print(f"❌ Failed to load {model_key}: {e}")

    print(f"\\nAll models loaded: {list(models.keys())}")
    yield
    models.clear()


app = FastAPI(
    title="Fun-ASR Batch-Optimized API",
    description="支持Nano和MLT双模型，内置batch处理优化"
)


def get_model(model_key: Optional[str] = None):
    """获取模型"""
    if model_key is None:
        model_key = "nano"

    if model_key not in models:
        available = list(models.keys())
        if not available:
            raise HTTPException(status_code=503, detail="No models loaded")
        model_key = available[0]

    return models[model_key]["model"], models[model_key]["kwargs"], model_key


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(default="auto"),
    model: Optional[str] = Form(default=None),
    hotwords: str = Form(default=""),
    itn: bool = Form(default=True),
):
    """单文件语音转文字"""
    asr_model, kwargs, model_used = get_model(model)

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        hw_list = [w.strip() for w in hotwords.split(",") if w.strip()] if hotwords else []
        lang = language if language != "auto" else ("zh" if model_used == "mlt" else "中文")

        with torch.inference_mode():
            # 使用inference方法（batch优化支持）
            results = asr_model.inference(
                data_in=[tmp_path],
                batch_size=1,
                language=lang,
                hotwords=hw_list if hw_list else " ",  # FunASR需要非空hotwords
                **kwargs
            )

        # 提取文本结果
        if results and len(results) > 0 and len(results[0]) > 0:
            text = results[0][0].get("text", "") if isinstance(results[0][0], dict) else ""
        else:
            text = ""

        return {
            "text": text,
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
    model: Optional[str] = Form(default=None),
    hotwords: str = Form(default=""),
    itn: bool = Form(default=True),
):
    """批量语音转文字 - Batch优化版本"""
    asr_model, kwargs, model_used = get_model(model)

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
        lang = language if language != "auto" else ("zh" if model_used == "mlt" else "中文")

        batch_size = min(len(tmp_paths), MAX_BATCH_SIZE)

        with torch.inference_mode():
            # 使用inference方法进行batch处理
            results = asr_model.inference(
                data_in=tmp_paths,
                batch_size=batch_size,  # 关键：启用batch处理
                language=lang,
                hotwords=hw_list if hw_list else " ",
                **kwargs
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
            "model": model_used,
            "batch_size": batch_size,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
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
        "batch_optimized": True
    }


@app.get("/models")
async def list_models():
    """列出所有可用模型"""
    result = {}
    for key, config in MODEL_CONFIGS.items():
        result[key] = {
            **config,
            "loaded": key in models,
            "batch_optimized": True
        }
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
