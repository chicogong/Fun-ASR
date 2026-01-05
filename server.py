#!/usr/bin/env python3
"""
Fun-ASR API服务器
提供RESTful API接口进行语音识别
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import torch
import tempfile
import os
import time
import traceback
from pathlib import Path
import logging
from typing import List

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(
    title="Fun-ASR API",
    description="高精度多语言语音识别服务",
    version="1.0.0"
)

# 全局模型变量
global_model = None
model_status = "未初始化"

def get_device():
    """获取最佳计算设备"""
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_asr_model():
    """加载ASR模型"""
    global global_model, model_status

    try:
        device = "cpu"  # 强制使用CPU避免CUDA依赖

        # 尝试可用的模型
        models_to_try = [
            "damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "damo/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch",
        ]

        from funasr import AutoModel

        for model_dir in models_to_try:
            try:
                logger.info(f"尝试加载模型: {model_dir} 到设备: {device}")

                model = AutoModel(
                    model=model_dir,
                    device=device,
                    disable_update=True
                )

                logger.info(f"✅ 模型加载成功: {model_dir}")
                break
            except Exception as e:
                logger.error(f"❌ 模型 {model_dir} 加载失败: {e}")
                continue
        else:
            raise Exception("所有模型都加载失败")

        global_model = model
        model_status = f"模型加载成功 - 设备: {device}"
        logger.info(model_status)
        return True

    except Exception as e:
        error_msg = f"模型加载失败: {str(e)}"
        model_status = error_msg
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False

@app.on_event("startup")
def startup_event():
    """应用启动时加载模型"""
    logger.info("启动Fun-ASR API服务...")
    load_asr_model()

@app.get("/health")
async def health_check():
    """健康检查接口"""
    device = get_device()
    cuda_available = torch.cuda.is_available()

    models_loaded = []
    if global_model is not None:
        models_loaded.append("nano")

    return {
        "status": "ok" if global_model is not None else "error",
        "model_status": model_status,
        "device": device,
        "cuda_available": cuda_available,
        "models_loaded": models_loaded,
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """根路径信息"""
    return {
        "service": "Fun-ASR API",
        "version": "1.0.0",
        "description": "高精度多语言语音识别服务",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = "中文",
    itn: bool = True,
    hotwords: str = ""
):
    """
    语音转文字接口

    Args:
        file: 音频文件 (支持 mp3, wav, m4a, flac)
        language: 识别语言 (中文/English/日本語等)
        itn: 是否进行逆文本规范化
        hotwords: 热词列表 (逗号分隔)

    Returns:
        转录结果和相关信息
    """
    if global_model is None:
        raise HTTPException(status_code=503, detail="模型未加载或加载失败")

    # 检查文件格式
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.aac')):
        raise HTTPException(status_code=400, detail="不支持的音频格式")

    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # 处理热词
        hotwords_list = [w.strip() for w in hotwords.split(",") if w.strip()] if hotwords else []

        logger.info(f"转录文件: {file.filename}, 语言: {language}")

        start_time = time.time()

        # 执行语音识别
        result = global_model.generate(
            input=[temp_file_path],
            cache={},
            batch_size=1,
            language=language,
            itn=itn,
            hotwords=hotwords_list
        )

        end_time = time.time()
        processing_time = end_time - start_time

        # 清理临时文件
        try:
            os.unlink(temp_file_path)
        except:
            pass

        # 返回结果
        text = result[0]["text"]

        return {
            "success": True,
            "text": text,
            "language": language,
            "processing_time": round(processing_time, 2),
            "file_name": file.filename,
            "file_size": len(content),
            "device": get_device(),
            "itn": itn,
            "hotwords": hotwords_list
        }

    except Exception as e:
        # 清理临时文件
        try:
            os.unlink(temp_file_path)
        except:
            pass

        error_msg = f"转录失败: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/transcribe/batch")
async def transcribe_batch(
    files: List[UploadFile] = File(...),
    language: str = "中文",
    itn: bool = True
):
    """
    批量语音转文字接口
    """
    if global_model is None:
        raise HTTPException(status_code=503, detail="模型未加载或加载失败")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="批量处理最多支持10个文件")

    results = []

    for file in files:
        try:
            # 处理单个文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name

            start_time = time.time()

            result = global_model.generate(
                input=[temp_file_path],
                cache={},
                batch_size=1,
                language=language,
                itn=itn
            )

            end_time = time.time()

            # 清理临时文件
            try:
                os.unlink(temp_file_path)
            except:
                pass

            text = result[0]["text"]

            results.append({
                "file_name": file.filename,
                "text": text,
                "processing_time": round(end_time - start_time, 2),
                "success": True
            })

        except Exception as e:
            results.append({
                "file_name": file.filename,
                "error": str(e),
                "success": False
            })

    return {
        "results": results,
        "total_files": len(files),
        "successful": len([r for r in results if r["success"]]),
        "language": language
    }

@app.get("/models")
async def get_models():
    """获取可用模型信息"""
    models = []

    if global_model is not None:
        models.append({
            "name": "Fun-ASR-Nano-2512",
            "status": "loaded",
            "languages": ["中文", "English", "日本語"],
            "features": ["多方言", "地方口音", "音乐背景", "低延迟"]
        })

    return {
        "models": models,
        "total": len(models)
    }

@app.get("/languages")
async def get_supported_languages():
    """获取支持的语言列表"""
    return {
        "languages": [
            {"code": "zh", "name": "中文", "dialects": ["吴语", "粤语", "闽语", "客家话", "赣语", "湘语", "晋语"]},
            {"code": "en", "name": "English", "dialects": []},
            {"code": "ja", "name": "日本語", "dialects": []},
            {"code": "ko", "name": "한국어", "dialects": []},
            {"code": "vi", "name": "Tiếng Việt", "dialects": []},
            {"code": "th", "name": "ไทย", "dialects": []},
            {"code": "ar", "name": "العربية", "dialects": []}
        ]
    }

if __name__ == "__main__":
    logger.info("启动Fun-ASR API服务器...")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )