import hashlib
import collections
import os
import typing

import librosa
import omegaconf
import soundfile as sf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, CsmForConditionalGeneration
from xcodec2.modeling_xcodec2 import XCodec2Model

from src.enums.engine_type import EngineType
from src.utils.filesystem_utils import get_app_root
from src.config.config import cfg
from src.tts_engines.tts_engine import tts_engine
from src.utils.audio_utils import load_audio

import torchaudio

generator = load_csm_1b(device="cuda")


def string_to_int(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (10 ** 8)


class CSM1BEngine(tts_engine):

    def __init__(self):
        super().__init__()
        print("Setting Up CSM1B Engine")
        self.engin_type = EngineType.CSM1B
        self.engine_name = self.engin_type.value
        self.device = cfg.get(cfg.device)
        self.processor = None
        self.tokenize = None

    def generate_audio(self, text, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        # Get audio data and sample rate from inference
        audio_data, sample_rate = self.inference(text, transcript, voice, language, output_file, streaming)
        self.process_audio(audio_data, sample_rate, output_file)

    def load_model(self):
        print("Loading Llasa Model")

        if self.is_base:
            self.processor = AutoProcessor.from_pretrained(
                str(os.path.abspath(os.path.join(get_app_root(), 'Sesame', 'csm-1b'))))
            self.model = CsmForConditionalGeneration.from_pretrained(
                str(os.path.abspath(os.path.join(get_app_root(), 'Sesame', 'csm-1b'))), device_map=self.device)
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = CsmForConditionalGeneration.from_pretrained(self.model_path, device_map=self.device)

        if self.device == 'cpu':
            self.model.eval().cpu()
        else:
            self.model.eval()

    def unload_model(self):
        self.basic_unload_model()
        del self.processor
        self.processor = None

    @torch.no_grad()
    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False,
                  speaker=None):
        speaker_id = "0" if speaker is None else string_to_int(speaker)
        conversation = []

        if voice is not None and transcript is not None:
            conversation.append({
                "role": f"{speaker_id}",
                "content": [{"type": "text", "text": transcript}, {"type": "audio", "path": load_audio(voice, 24000)}],
            })

        conversation.append(
            {
                "role": f"{speaker_id}",
                "content": [{"type": "text", "text": text}]
            },
        )
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(self.device)

        # infer the model
        audio_values = self.model.generate(**inputs, output_audio=True)
        return audio_values[0].to(torch.float32).cpu().numpy(), 24000
