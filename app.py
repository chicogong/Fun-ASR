#!/usr/bin/env python3
"""
Fun-ASR Web服务
基于Gradio的语音识别Web界面
"""

import gradio as gr
import torch
import os
import tempfile
import traceback
import time
from pathlib import Path

def check_system_info():
    """检查系统信息"""
    info = {
        "device": "cpu",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "model_status": "未加载"
    }

    if torch.cuda.is_available():
        info["device"] = "cuda:0"
        info["gpu_name"] = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        info["device"] = "mps"

    return info

def load_asr_model():
    """加载ASR模型"""
    try:
        device = check_system_info()["device"]
        model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"

        # 尝试使用FunASR
        from funasr import AutoModel

        model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            remote_code="./model.py",
            device=device,
            disable_update=True
        )
        return model, "success", f"模型加载成功，使用设备: {device}"

    except Exception as e:
        return None, "error", f"模型加载失败: {str(e)}"

# 全局模型变量
global_model = None
model_status = "未初始化"

def transcribe_audio(audio_file):
    """语音转文字"""
    global global_model, model_status

    if audio_file is None:
        return "❌ 请上传音频文件"

    try:
        # 如果模型未加载，尝试加载
        if global_model is None:
            model_status = "正在加载模型..."
            global_model, status, msg = load_asr_model()
            model_status = msg

            if status != "success":
                return f"❌ {msg}"

        start_time = time.time()

        # 执行推理
        result = global_model.generate(
            input=[audio_file],
            cache={},
            batch_size=1,
            language="中文",
            itn=True
        )

        end_time = time.time()
        processing_time = end_time - start_time

        text = result[0]["text"]

        return f"""✅ 识别成功！

🎯 识别结果: {text}
⏱️ 处理时间: {processing_time:.2f}秒
💻 设备: {check_system_info()['device']}
📊 模型状态: {model_status}"""

    except Exception as e:
        error_msg = f"❌ 识别失败: {str(e)}"
        if "模型未注册" in str(e) or "not registered" in str(e):
            error_msg += "\n\n💡 建议:\n1. 检查网络连接\n2. 等待模型下载完成\n3. 重试几次"
        return error_msg

def get_system_info_display():
    """获取系统信息显示"""
    info = check_system_info()

    display_text = f"""
🖥️ **系统信息**
- PyTorch版本: {info['torch_version']}
- 计算设备: {info['device']}
- CUDA可用: {'✅' if info['cuda_available'] else '❌'}
- MPS可用: {'✅' if info['mps_available'] else '❌'}

🤖 **模型状态**
- 状态: {model_status}

🎯 **功能特性**
- 支持中文、英文、日文等多种语言
- 支持7种中文方言识别
- 支持26种地方口音
- 支持音乐背景下的歌词识别
"""

    if info['cuda_available']:
        display_text += f"\n🚀 **GPU信息**\n- GPU: {info.get('gpu_name', '未知')}"

    return display_text

def create_demo_interface():
    """创建演示界面"""

    # 自定义CSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    """

    with gr.Blocks(css=css, title="Fun-ASR 语音识别服务") as demo:

        gr.HTML('<h1 class="main-header">🎯 Fun-ASR 语音识别服务</h1>')
        gr.HTML('<p class="subtitle">高精度多语言语音识别 | 支持31种语言 | 低延迟实时转录</p>')

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📤 上传音频文件")
                audio_input = gr.Audio(
                    label="选择音频文件",
                    type="filepath",
                    sources=["upload", "microphone"]
                )

                with gr.Row():
                    transcribe_btn = gr.Button(
                        "🎯 开始识别",
                        variant="primary",
                        size="lg"
                    )
                    clear_btn = gr.Button("🗑️ 清除", size="lg")

            with gr.Column(scale=3):
                gr.Markdown("### 📝 识别结果")
                output_text = gr.Textbox(
                    label="转录结果",
                    lines=10,
                    max_lines=20,
                    placeholder="识别结果将显示在这里..."
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💻 系统信息")
                info_btn = gr.Button("🔍 查看系统状态")
                info_output = gr.Textbox(
                    label="系统状态",
                    lines=15,
                    value=get_system_info_display(),
                    interactive=False
                )

        with gr.Row():
            gr.Markdown("""
### 📚 使用说明

1. **上传音频**: 点击上传区域选择音频文件，支持 MP3、WAV、M4A 等格式
2. **录制音频**: 点击麦克风图标直接录制语音
3. **开始识别**: 点击"开始识别"按钮进行语音转文字
4. **查看结果**: 识别结果会显示在右侧文本框中

### 🌍 支持语言

**主要语言**: 中文、English、日本語、한국어、Tiếng Việt、ไทย、العربية

**中文方言**: 吴语、粤语、闽语、客家话、赣语、湘语、晋语

**地方口音**: 河南、陕西、湖北、四川、重庆、云南、贵州、广东、广西等26个地区

### ⚡ 性能特点

- **高精度**: 远场高噪声环境识别准确率达93%
- **低延迟**: 支持实时转录
- **多场景**: 会议室、车载、工业现场等
- **音乐背景**: 支持音乐背景下的歌词识别
            """)

        # 事件绑定
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input],
            outputs=[output_text]
        )

        clear_btn.click(
            fn=lambda: (None, ""),
            outputs=[audio_input, output_text]
        )

        info_btn.click(
            fn=get_system_info_display,
            outputs=[info_output]
        )

    return demo

def main():
    """主函数"""
    print("🚀 启动Fun-ASR Web服务...")
    print("=" * 50)

    # 检查系统信息
    sys_info = check_system_info()
    print(f"📊 系统信息:")
    print(f"  - PyTorch: {sys_info['torch_version']}")
    print(f"  - 设备: {sys_info['device']}")
    print(f"  - CUDA: {sys_info['cuda_available']}")
    print(f"  - MPS: {sys_info['mps_available']}")
    print()

    # 创建界面
    demo = create_demo_interface()

    print("🌐 启动Web界面...")

    # 启动服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 本地部署不需要公共链接
        show_error=True,
        inbrowser=False  # 不自动打开浏览器
    )

if __name__ == "__main__":
    main()