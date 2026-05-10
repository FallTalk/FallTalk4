import os
import uuid

import soundfile as sf
import torch

from omnivoice import OmniVoice

from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.tts_engines.tts_engine import tts_engine
from src.utils.audio_utils import load_audio
from src.utils.filesystem_utils import get_app_root


def _resolve_device_map(device: str) -> str:
    if device.startswith("cuda"):
        return device if ":" in device else "cuda:0"
    return device


class OmniVoiceEngine(tts_engine):
    def __init__(self):
        super().__init__()
        self.engine_type = EngineType.OMNIVOICE
        self.engine_name = self.engine_type.value
        self.device = cfg.get(cfg.device)
        self.model = None

    def load_model(self):
        local_path = os.path.join(get_app_root(), "models", "OmniVoice")
        model_id = local_path if os.path.isdir(local_path) else "k2-fsa/OmniVoice"
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model_name = "OmniVoice"
        self.model_path = model_id

        self.model = OmniVoice.from_pretrained(
            model_id,
            device_map=_resolve_device_map(self.device),
            dtype=dtype,
        )

    def unload_model(self):
        self.basic_unload_model()
        self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False, speaker=None, start_time=None, end_time=None):
        if self.model is None:
            self.load_model()

        num_step = cfg.get(cfg.omnivoice_num_step)
        speed = max(0.1, cfg.get(cfg.omnivoice_speed) / 100.0)
        instruct = (cfg.get(cfg.omnivoice_instruct) or "").strip()
        ref_text = None

        if isinstance(transcript, dict):
            ref_text = transcript.get("transcript")
        elif transcript:
            ref_text = transcript

        generate_kwargs = {
            "text": text,
            "num_step": num_step,
            "speed": speed,
        }

        resampled_voice = None
        if voice and os.path.exists(voice):
            resampled_voice = os.path.join(get_app_root(), "temp", f"{uuid.uuid4().hex}_24k.wav")
            audio_data = load_audio(voice, 24000)
            sf.write(resampled_voice, audio_data, 24000)
            generate_kwargs["ref_audio"] = resampled_voice
            if ref_text:
                generate_kwargs["ref_text"] = ref_text
        elif instruct:
            generate_kwargs["instruct"] = instruct

        try:
            audio = self.model.generate(**generate_kwargs)
        finally:
            if resampled_voice and os.path.exists(resampled_voice):
                os.remove(resampled_voice)

        return audio[0], 24000
