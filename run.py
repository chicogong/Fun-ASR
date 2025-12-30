"""Fun-ASR-Nano 推理脚本"""

import argparse
import torch
from funasr import AutoModel
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fun-ASR-Nano 语音识别")
    parser.add_argument("--audio", "-a", required=True, help="音频文件路径")
    parser.add_argument("--model", "-m", default="FunAudioLLM/Fun-ASR-Nano-2512", help="模型路径")
    parser.add_argument("--language", "-l", default="中文", help="语言")
    parser.add_argument("--device", "-d", default=None, help="设备 (cuda:0/cpu)")
    parser.add_argument("--hotwords", nargs="+", default=[], help="热词列表")
    parser.add_argument("--no-itn", action="store_true", help="禁用文本规整")
    args = parser.parse_args()

    # 设备选择
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # 加载模型
    model = AutoModel(
        model=args.model,
        trust_remote_code=True,
        remote_code="./model.py",
        device=device,
        disable_update=True,
    )

    # 推理
    with torch.inference_mode():
        result = model.generate(
            input=[args.audio],
            cache={},
            batch_size=1,
            language=args.language,
            hotwords=args.hotwords if args.hotwords else [],
            itn=not args.no_itn,
        )

    print(result[0]["text"])


if __name__ == "__main__":
    main()
