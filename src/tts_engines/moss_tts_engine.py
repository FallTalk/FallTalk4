import os
from importlib.util import find_spec

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.tts_engines.tts_engine import tts_engine
from src.utils.filesystem_utils import get_app_root


def _resolve_device(device: str) -> str:
    if device.startswith("cuda"):
        return device if ":" in device else "cuda:0"
    return device


def _resolve_attn(device: str, dtype: torch.dtype) -> str:
    if device.startswith("cuda"):
        if find_spec("flash_attn") is not None and dtype in {torch.float16, torch.bfloat16}:
            return "flash_attention_2"
        return "sdpa"
    return "eager"


class MossTTSEngine(tts_engine):
    def __init__(self):
        super().__init__()
        self.engine_type = EngineType.MOSS_TTS
        self.engine_name = self.engine_type.value
        self.device = cfg.get(cfg.device)
        self.processor = None

    def load_model(self):
        local_path = os.path.join(get_app_root(), "models", "MOSS-TTS")
        model_id = local_path if os.path.isdir(local_path) else "OpenMOSS-Team/MOSS-TTS"
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        attn_implementation = _resolve_attn(self.device, dtype)
        self.model_name = "MOSS-TTS"
        self.model_path = model_id

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(_resolve_device(self.device))
        self.processor.audio_tokenizer.eval()

        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        ).to(_resolve_device(self.device))
        self.model.eval()

    def unload_model(self):
        self.basic_unload_model()
        self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_reference_audio(self, voice: str):
        waveform, sample_rate = torchaudio.load(voice)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, sample_rate

    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False, speaker=None, start_time=None, end_time=None):
        if self.model is None or self.processor is None:
            self.load_model()

        device = _resolve_device(self.device)
        target_sr = int(self.processor.model_config.sampling_rate)

        if voice and os.path.exists(voice):
            waveform, sample_rate = self._load_reference_audio(voice)
            if sample_rate != target_sr:
                waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
            reference_audio_codes = self.processor.encode_audios_from_wav([waveform], sampling_rate=target_sr)
            conversations = [[self.processor.build_user_message(text=text, reference=reference_audio_codes)]]
        else:
            conversations = [[self.processor.build_user_message(text=text)]]

        batch = self.processor(conversations, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg.get(cfg.moss_max_new_tokens),
                temperature=cfg.get(cfg.moss_temperature) / 100.0,
                top_p=cfg.get(cfg.moss_top_p) / 100.0,
                top_k=cfg.get(cfg.moss_top_k),
                do_sample=True,
            )

        message = self.processor.decode(outputs)[0]
        audio = message.audio_codes_list[0]
        return audio.detach().cpu().float().numpy(), target_sr
