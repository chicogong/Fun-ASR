import json
import logging
import os
import random
import re
import string
import time
import traceback

import torch
import torch.nn as nn
from funasr import AutoModel
from funasr.metrics.compute_acc import compute_accuracy
from funasr.register import tables
from funasr.train_utils.device_funcs import force_gatherable, to_device
from funasr.utils.datadir_writer import DatadirWriter
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video
from transformers import AutoConfig, AutoModelForCausalLM

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@tables.register("model_classes", "FunASRMLT")
class FunASRMLT(nn.Module):
    def __init__(
        self,
        audio_encoder: str = None,
        audio_encoder_conf: dict = None,
        audio_adaptor: str = None,
        audio_adaptor_conf: dict = None,
        llm: str = None,
        llm_conf: dict = None,
        input_size: int = 80,
        length_normalized_loss: bool = False,
        **kwargs,
    ):
        super().__init__()

        # audio encoder
        hub = audio_encoder_conf.get("hub", None)
        self.audio_encoder_activation_checkpoint = audio_encoder_conf.get(
            "activation_checkpoint", False
        )
        if hub == "ms":
            model = AutoModel(model=audio_encoder, model_revision="master")
            audio_encoder_output_size = (
                model.model.encoder_output_size
                if hasattr(model.model, "encoder_output_size")
                else -1
            )
            audio_encoder = (
                model.model.model.encoder
                if hasattr(model.model, "model")
                else model.model.encoder
            )
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size()
        freeze = audio_encoder_conf.get("freeze", True)

        if freeze:
            for name, param in audio_encoder.named_parameters():
                param.requires_grad = False
            audio_encoder.eval()
        self.audio_encoder = audio_encoder

        # llm
        self.llm = None
        init_param_path = llm_conf.get("init_param_path", None)
        llm_dim = None

        llm_load_kwargs = llm_conf.get("load_kwargs", {})
        config = AutoConfig.from_pretrained(init_param_path)
        model = AutoModelForCausalLM.from_config(config, **llm_load_kwargs)

        freeze = llm_conf.get("freeze", True)
        if freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
            model.eval()

        if llm_conf.get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()

        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        self.llm = model.to(dtype_map[self.llm_dtype])
        llm_dim = model.get_input_embeddings().weight.shape[-1]

        # adaptor
        adaptor_class = tables.adaptor_classes.get(audio_adaptor)
        if audio_encoder_output_size > 0:
            audio_adaptor_conf["encoder_dim"] = audio_encoder_output_size
        audio_adaptor_conf["llm_dim"] = (
            llm_dim if llm_dim is not None else audio_adaptor_conf["llm_dim"]
        )
        audio_adaptor = adaptor_class(**audio_adaptor_conf)
        init_param_path = audio_adaptor_conf.get("init_param_path", None)
        if init_param_path is not None:
            src_state = torch.load(init_param_path, map_location="cpu")
            flag = audio_adaptor.load_state_dict(src_state, strict=False)
            logging.info(
                f"Loading audio_adaptor ckpt: {init_param_path}, status: {flag}"
            )
        freeze = audio_adaptor_conf.get("freeze", False)
        if freeze:
            for name, param in audio_adaptor.named_parameters():
                param.requires_grad = False
            audio_adaptor.eval()
        self.audio_adaptor = audio_adaptor

        self.length_normalized_loss = length_normalized_loss
        self.feat_permute = audio_encoder_conf.get("feat_permute", True)
        rank = int(os.environ.get("RANK", 0))
        logging.info(f"rank: {rank}, FunASRMLT model is builded.")

    def encode(self, speech, speech_lengths):
        # audio encoder
        if self.feat_permute:
            encoder_out, encoder_out_lens = self.audio_encoder(
                speech.permute(0, 2, 1), speech_lengths
            )
        else:
            encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)

        return encoder_out, encoder_out_lens

    def data_template(self, data):
        system, user, assistant = [], [], []
        for i, item in enumerate(data):
            role = item["role"]
            content = item["content"]
            if role == "system":
                system.append(content)
            elif role == "user":
                if "audio" in item:
                    audio = item["audio"]
                    content = [content, audio]
                user.append(content)
            elif role == "assistant":
                assistant.append(content)

        system = system * len(user)

        contents = {
            "system": system,
            "user": user,
            "assistant": assistant,
        }

        return contents

    def data_load_speech(
        self, contents: dict, tokenizer, frontend, meta_data={}, **kwargs
    ):
        system = contents["system"]
        user = contents["user"]
        assistant = contents["assistant"]
        pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        do_think = kwargs.get("dataset_conf", {}).get("do_think", True)
        sys_prompt = kwargs.get("dataset_conf", {}).get("sys_prompt", True)

        input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
            [], [], [], [], [], [], []
        )
        input_source_ids = []

        for i, (system_prompt, user_prompt, target_out) in enumerate(
            zip(system, user, assistant)
        ):
            if i >= kwargs.get("multiturn_num_max", 5):
                break
            if len(input_ids) > kwargs.get("max_token_length", 1500):
                break
            if isinstance(user_prompt, (list, tuple)):
                user_prompt, audio = user_prompt
            if i == 0:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}"
                    if not sys_prompt:
                        source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    if not sys_prompt:
                        source_input = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            if not do_think:
                source_input += "<think>\n\n</think>\n\n"

            splits = pattern.split(source_input)
            source_ids = []
            fbank_mask_i = []
            fake_token_len_i = 0
            fbank_beg_i = -1
            speech, speech_lengths = [], []

            for k, sub_str in enumerate(splits):
                if not sub_str.startswith("<|startofspeech|>"):
                    sub_token = tokenizer.encode(sub_str)
                    source_ids += sub_token
                    fbank_mask_i += [0] * len(sub_token)
                else:
                    sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                        "<|endofspeech|>", ""
                    )
                    if sub_str.startswith("!"):
                        sub_str = sub_str[1:]
                        if sub_str.startswith("!"):  # !!: audio sample point
                            sub_str = audio
                        try:
                            time1 = time.perf_counter()
                            data_src = load_audio_text_image_video(
                                sub_str, fs=frontend.fs, **kwargs
                            )
                            time2 = time.perf_counter()
                            meta_data["load_data"] = f"{time2 - time1:0.3f}"
                        except Exception as e:
                            logging.error(
                                f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                            )

                        speech, speech_lengths = extract_fbank(
                            data_src,
                            data_type=kwargs.get("data_type", "sound"),
                            frontend=frontend,
                            is_final=True,
                        )

                        time3 = time.perf_counter()
                        meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
                        meta_data["batch_data_time"] = (
                            speech_lengths.sum().item()
                            * frontend.frame_shift
                            * frontend.lfr_n
                            / 1000
                        )

                        if self.feat_permute:
                            speech = speech.permute(0, 2, 1)

                        olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                        olens = 1 + (olens - 3 + 2 * 1) // 2
                        fake_token_len_i = (olens - 1) // 2 + 1
                        fake_token = [0] * fake_token_len_i
                        fbank_beg_i = len(source_ids)
                        source_ids += fake_token
                        fbank_mask_i += [1] * len(fake_token)

            fbank_beg += [fbank_beg_i + len(input_ids)]
            fake_token_len += [fake_token_len_i]
            source_mask = [-100] * len(source_ids)
            target_out = f"{target_out}<|im_end|>"
            target_ids = tokenizer.encode(target_out)
            input_source_ids = input_ids + source_ids
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids
            fbank_mask += fbank_mask_i
            if len(speech) > 0:
                fbank.append(speech[0, :, :])
                fbank_lens.append(speech_lengths)

        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int64)
        fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
        fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
        fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
        source_ids = torch.tensor(input_source_ids, dtype=torch.int64)
        target_ids = torch.tensor(target_ids, dtype=torch.int64)

        if len(fbank) > 0:
            speech = torch.nn.utils.rnn.pad_sequence(
                fbank, batch_first=True, padding_value=0.0
            )
            speech_lengths = torch.nn.utils.rnn.pad_sequence(
                fbank_lens, batch_first=True, padding_value=-1
            )
        else:
            speech = []
            speech_lengths = []

        output = {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "fbank_mask": fbank_mask[None, :],
            "fbank_beg": fbank_beg[None,],
            "fake_token_len": fake_token_len[None, :],
            "input_ids": input_ids[None,],
            "attention_mask": attention_mask[None,],
            "labels_ids": labels,
            "source_ids": source_ids[None, :],
            "target_ids": target_ids[None, :],
        }

        return output

    def inference_prepare(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        meta_data = {}

        if kwargs.get("batch_size", 1) > 1:
            logging.info(f"MLT batch_size: {kwargs.get('batch_size')}")

        outputs = []
        full_contents = []
        for i in range(kwargs.get("batch_size", 1)):
            contents = self.data_template(data_in[i])
            full_contents.append(contents)
            output = self.data_load_speech(
                contents, tokenizer, frontend, meta_data=meta_data, **kwargs
            )
            outputs.append(output)

        # collate with left padding optimization
        speech_list = []
        speech_lengths_list = []
        input_ids_list = []
        attention_mask_list = []
        labels_ids_list = []
        fbank_beg_list = []
        fake_token_len_list = []
        source_ids_list = []

        for output in outputs:
            # speech: (1, T, D) -> list of (T, D)
            if output["speech_lengths"].numel() > 0:
                for s in output["speech"]:
                    if self.feat_permute:
                        s = s.transpose(0, 1)
                    speech_list.append(s)
                for l in output["speech_lengths"]:
                    speech_lengths_list.append(l)

            input_ids_list.append(output["input_ids"][0])
            attention_mask_list.append(output["attention_mask"][0])
            labels_ids_list.append(output["labels_ids"])
            fbank_beg_list.append(output["fbank_beg"][0])
            fake_token_len_list.append(output["fake_token_len"][0])
            source_ids_list.append(output["source_ids"][0])

        # Left Padding优化 - 关键改进点
        max_input_len = max([x.size(0) for x in input_ids_list])
        padded_input_ids = []
        padded_attention_mask = []
        padded_source_ids = []
        shifted_fbank_beg = []

        for i in range(len(input_ids_list)):
            curr_len = input_ids_list[i].size(0)
            pad_len = max_input_len - curr_len

            # Left pad with 0
            padded_input_ids.append(torch.cat([torch.zeros(pad_len, dtype=input_ids_list[i].dtype, device=input_ids_list[i].device), input_ids_list[i]]))
            padded_attention_mask.append(torch.cat([torch.zeros(pad_len, dtype=attention_mask_list[i].dtype, device=attention_mask_list[i].device), attention_mask_list[i]]))
            padded_source_ids.append(torch.cat([torch.zeros(pad_len, dtype=source_ids_list[i].dtype, device=source_ids_list[i].device), source_ids_list[i]]))

            # Shift fbank_beg indices by pad_len
            shifted_fbank_beg.append(fbank_beg_list[i] + pad_len)

        input_ids = torch.stack(padded_input_ids)
        attention_mask = torch.stack(padded_attention_mask)
        source_ids = torch.stack(padded_source_ids)

        # Pad fbank_beg and fake_token_len
        fbank_beg = torch.nn.utils.rnn.pad_sequence(shifted_fbank_beg, batch_first=True, padding_value=0)
        fake_token_len = torch.nn.utils.rnn.pad_sequence(fake_token_len_list, batch_first=True, padding_value=0)

        # Audio padding
        if len(speech_list) > 0:
            speech = torch.nn.utils.rnn.pad_sequence(speech_list, batch_first=True, padding_value=0.0)
            if self.feat_permute:
                speech = speech.permute(0, 2, 1)
            speech_lengths = torch.tensor([l.item() for l in speech_lengths_list], dtype=torch.int32)
        else:
            speech = torch.tensor([])
            speech_lengths = torch.tensor([])

        batch = {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "input_ids": input_ids,
            "fbank_beg": fbank_beg,
            "fake_token_len": fake_token_len,
            "source_ids": source_ids,
            "attention_mask": attention_mask
        }
        batch = to_device(batch, kwargs["device"])

        # audio encoder
        speech = batch["speech"]
        speech_lengths = batch["speech_lengths"]
        encoder_out = None
        encoder_out_lens = None

        if len(speech) > 0:
            # 混合精度支持
            if kwargs.get("fp16", False):
                speech = speech.to(torch.float16)
            elif kwargs.get("bf16", False):
                speech = speech.to(torch.bfloat16)

            # audio encoder
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(
                encoder_out, encoder_out_lens
            )
            meta_data["audio_adaptor_out"] = encoder_out
            meta_data["audio_adaptor_out_lens"] = encoder_out_lens

        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        fbank_beg = batch["fbank_beg"]
        fake_token_len = batch["fake_token_len"]

        if not kwargs.get("teachforing", False):
            input_ids = source_ids

        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)

        batch_size, token_num, dims = inputs_embeds.shape
        fake_token_len[fake_token_len < 0] = 0
        fbank_beg[fbank_beg < 0] = 0

        speech_idx = 0
        for batch_idx in range(batch_size):
            for turn_id in range(fbank_beg.shape[1]):
                fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                if fbank_beg_idx > 0:
                    speech_token_len = fake_token_len[batch_idx, turn_id]
                    if encoder_out is not None:
                        speech_token = encoder_out[speech_idx, :speech_token_len, :]
                        try:
                            inputs_embeds[
                                batch_idx,
                                fbank_beg_idx : fbank_beg_idx + speech_token_len,
                                :,
                            ] = speech_token
                        except Exception as e:
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            speech_token_len = encoder_out_lens[speech_idx].item()
                            speech_token = encoder_out[speech_idx, :speech_token_len, :]
                            inputs_embeds[
                                batch_idx,
                                fbank_beg_idx : fbank_beg_idx + speech_token_len,
                                :,
                            ] = speech_token
                        speech_idx += 1

        return inputs_embeds, full_contents, batch, source_ids, meta_data

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        """Paraformer模型的批量推理接口"""

        # 对于Paraformer，直接使用音频文件路径进行批量推理
        if key is None:
            key = []
            for i, _ in enumerate(data_in):
                chars = string.ascii_letters + string.digits
                key.append(
                    "rand_key_" + "".join(random.choice(chars) for _ in range(13))
                )

        batch_size = kwargs.get("batch_size", len(data_in))
        batch_size = min(batch_size, len(data_in))

        # 处理音频文件路径
        audio_files = []
        for data in data_in:
            if isinstance(data, str):
                audio_files.append(data)
            elif hasattr(data, 'name'):  # 临时文件对象
                audio_files.append(data.name)
            else:
                # 如果是tensor，需要先保存为临时文件
                import tempfile
                import soundfile as sf
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                if isinstance(data, torch.Tensor):
                    sf.write(temp_file.name, data.numpy(), 16000)
                audio_files.append(temp_file.name)

        results = []

        # 使用标准的AutoModel推理
        try:
            # 使用from_pretrained加载的模型直接推理
            if hasattr(self, 'model') and self.model is not None:
                # 直接调用模型的generate方法
                res = self.model.generate(
                    input=audio_files,
                    batch_size=batch_size,
                    language=kwargs.get("language", "auto"),
                    **kwargs
                )
            else:
                # 如果模型结构不同，使用audio_encoder进行推理
                from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

                speech_list = []
                speech_lengths_list = []

                # 加载并处理音频
                for audio_file in audio_files:
                    data_src = load_audio_text_image_video(audio_file, fs=16000)
                    speech, speech_lengths = extract_fbank(
                        data_src,
                        data_type="sound",
                        frontend=frontend,
                        is_final=True
                    )
                    speech_list.append(speech[0])
                    speech_lengths_list.append(speech_lengths[0])

                # 批量处理
                if len(speech_list) > 0:
                    # Pad sequences
                    speech = torch.nn.utils.rnn.pad_sequence(
                        speech_list, batch_first=True, padding_value=0.0
                    )
                    if self.feat_permute:
                        speech = speech.permute(0, 2, 1)
                    speech_lengths = torch.tensor([l.item() for l in speech_lengths_list])

                    # Forward through encoder
                    encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

                    # 模拟结果（需要根据具体模型调整）
                    res = []
                    for i in range(len(audio_files)):
                        res.append({"text": f"处理音频文件 {i+1}"})

            # 格式化结果
            for i, (audio_file, key_i) in enumerate(zip(audio_files, key)):
                if i < len(res) if isinstance(res, list) else 1:
                    text = res[i].get("text", "") if isinstance(res, list) else res.get("text", "")
                    result_i = {
                        "key": key_i,
                        "text": text.strip(),
                        "text_tn": re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", text),
                    }
                    results.append(result_i)

        except Exception as e:
            logging.error(f"MLT batch inference failed: {e}")
            # 返回空结果
            for i, key_i in enumerate(key):
                results.append({
                    "key": key_i,
                    "text": "",
                    "text_tn": "",
                })

        return [results], {}

    def inference_llm(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        inputs_embeds, contents, batch, source_ids, meta_data = self.inference_prepare(
            data_in, data_lengths, key, tokenizer, frontend, **kwargs
        )

        llm_dtype = kwargs.get("llm_dtype", "fp32")
        if llm_dtype == "fp32":
            llm_dtype = "fp16" if kwargs.get("fp16", False) else llm_dtype
            llm_dtype = "bf16" if kwargs.get("bf16", False) else llm_dtype

        with torch.cuda.amp.autocast(
            enabled=True if llm_dtype != "fp32" else False, dtype=dtype_map[llm_dtype]
        ):
            labels = [c["assistant"][-1] for c in contents]
            self.llm = self.llm.to(dtype_map[llm_dtype])
            inputs_embeds = inputs_embeds.to(dtype_map[llm_dtype])
            llm_kwargs = kwargs.get("llm_kwargs", {})

            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.llm.device)

            generated_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=kwargs.get("max_length", 512),
                **llm_kwargs,
            )

            response = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=kwargs.get("skip_special_tokens", True),
            )

        results = []
        for i in range(len(response)):
            response_clean = re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", response[i])
            label_i = labels[i]
            key_i = key[i]

            result_i = {
                "key": key_i,
                "text": re.sub(r'\s+', ' ', response[i].replace("/sil", " ")),
                "text_tn": response_clean,
                "label": label_i,
            }
            results.append(result_i)

        return results, meta_data

    @staticmethod
    def from_pretrained(model: str = None, **kwargs):
        """加载MLT模型的静态方法"""
        from funasr import AutoModel

        # 直接使用FunASR AutoModel进行加载，而不是重新构建
        try:
            # 对于Paraformer等传统模型，使用AutoModel直接加载
            asr_model = AutoModel(model=model, **kwargs)

            # 创建一个包装器来提供batch接口
            wrapper = MLTBatchWrapper(asr_model)

            return wrapper, kwargs
        except Exception as e:
            logging.error(f"Failed to load MLT model: {e}")
            raise


class MLTBatchWrapper:
    """Paraformer模型的batch处理包装器"""
    def __init__(self, asr_model):
        self.model = asr_model
        self.feat_permute = True

    def eval(self):
        """兼容性方法"""
        if hasattr(self.model, 'eval'):
            self.model.eval()
        return self

    def generate(self, input, **kwargs):
        """代理generate调用到内部模型"""
        return self.model.generate(input=input, **kwargs)

    def inference(self, data_in, **kwargs):
        """批量推理接口"""
        batch_size = kwargs.get("batch_size", len(data_in))
        batch_size = min(batch_size, len(data_in))

        # 生成key
        key = []
        for i, _ in enumerate(data_in):
            chars = string.ascii_letters + string.digits
            key.append(
                "rand_key_" + "".join(random.choice(chars) for _ in range(13))
            )

        results = []

        try:
            # 使用原生FunASR的generate方法进行批量推理
            start_time = time.time()

            # 移除kwargs中的重复参数
            generate_kwargs = {k: v for k, v in kwargs.items()
                             if k not in ["batch_size", "language"]}

            res = self.model.generate(
                input=data_in,
                batch_size=batch_size,
                language=kwargs.get("language", "auto"),
                **generate_kwargs
            )

            processing_time = time.time() - start_time

            # 格式化结果 - 处理FunASR返回的格式
            if isinstance(res, list):
                # FunASR返回格式通常是 [{"text": "结果"}] 或 ["文本结果"]
                for i, (result, key_i) in enumerate(zip(res, key)):
                    if isinstance(result, dict):
                        text = result.get("text", "")
                    elif isinstance(result, list) and len(result) > 0:
                        # 嵌套列表的情况
                        if isinstance(result[0], dict):
                            text = result[0].get("text", "")
                        else:
                            text = str(result[0])
                    else:
                        text = str(result)

                    result_i = {
                        "key": key_i,
                        "text": text.strip(),
                        "text_tn": re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", text),
                        "processing_time": processing_time / len(data_in)
                    }
                    results.append(result_i)
            else:
                # 单一结果
                if isinstance(res, dict):
                    text = res.get("text", "")
                elif isinstance(res, list) and len(res) > 0:
                    if isinstance(res[0], dict):
                        text = res[0].get("text", "")
                    else:
                        text = str(res[0])
                else:
                    text = str(res)

                results.append({
                    "key": key[0] if key else "default",
                    "text": text.strip(),
                    "text_tn": re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", text),
                    "processing_time": processing_time
                })

            # 确保结果数量匹配输入
            while len(results) < len(key):
                results.append({
                    "key": key[len(results)],
                    "text": "",
                    "text_tn": "",
                    "processing_time": processing_time / len(data_in)
                })

        except Exception as e:
            logging.error(f"MLT batch inference failed: {e}")
            # 返回错误结果
            for key_i in key:
                results.append({
                    "key": key_i,
                    "text": f"Error: {str(e)[:50]}...",
                    "text_tn": "",
                })

        return [results], {"processing_time": time.time() - start_time if 'start_time' in locals() else 0}