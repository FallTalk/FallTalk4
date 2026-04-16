import os
from typing import Optional

import librosa
import torch

from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.tts_engines.tts_engine import tts_engine
from src.utils import logging_utils
from src.utils.filesystem_utils import get_app_root

from dia.model import DEFAULT_SAMPLE_RATE
from transformers import AutoProcessor, DiaForConditionalGeneration, DiaProcessor


class DIA_Engine(tts_engine):

    def __init__(self):
        super().__init__()
        print("Setting Up DIA Engine")
        self.engine_type = EngineType.DIA
        self.engine_name = self.engine_type.value
        self.device = cfg.get(cfg.device)
        self.processor: Optional['DiaProcessor'] = None

    def clean(self):
        self.unload_model()
        del self.dia
        self.dia = None

    def unload_model(self):
        self.dia.dac_model.to('cpu')
        self.dia.model.to('cpu')
        super().basic_unload_model()


    def load_model(self):
        logging_utils.logger.debug(f"Loading {self.model_path}")
        if self.is_base:
            self.model = DiaForConditionalGeneration.from_pretrained(os.path.abspath(os.path.join(get_app_root(), 'models', 'DIA', '3b-0.1'))).to(self.device)
            self.processor = AutoProcessor.from_pretrained(os.path.abspath(os.path.join(get_app_root(), 'models', 'DIA', '3b-0.1')))
        else:
            self.model = DiaForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_path)


    @torch.no_grad()
    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False, speaker=None, start_time=None, end_time=None):
        transcript = "[S1] "+transcript
        input_text = [transcript + " " + text]

        audio, _ = librosa.load(voice, sr=DEFAULT_SAMPLE_RATE)

        inputs = self.processor(text=input_text, audio=audio, padding=True, return_tensors="pt").to(self.device)
        prompt_len = self.processor.get_audio_prompt_len(inputs["decoder_attention_mask"])
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        audio_outputs = self.processor.batch_decode(outputs, audio_prompt_len=prompt_len)
        return audio_outputs[0].cpu().float().numpy(), DEFAULT_SAMPLE_RATE
