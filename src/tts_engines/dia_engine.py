import os
import sys
from typing import Optional

import torch

from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.tts_engines.tts_engine import tts_engine
from src.utils import logging_utils
from src.utils import torch_utils
from src.utils.filesystem_utils import get_app_root, get_app_code_root

sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'dia/dia')))

from third_party.dia.dia.model import Dia, DEFAULT_SAMPLE_RATE

class DIA_Engine(tts_engine):

    def __init__(self):
        super().__init__()
        print("Setting Up DIA Engine")
        self.engin_type = EngineType.DIA
        self.engine_name = self.engin_type.value
        self.device = cfg.get(cfg.device)
        self.dia: Optional['Dia'] = None

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
            self.dia = Dia.from_pretrained(os.path.abspath(os.path.join(get_app_root(), 'models', 'DIA', '0.1')), compute_dtype=torch_utils.get_compute_dtype(), device=self.device)
        else:
            self.dia = Dia.from_pretrained(os.path.abspath(self.model_path), compute_dtype=str(torch_utils.get_compute_dtype()), device=self.device)

        self.model = self.dia.model
        self.dia.model = self.dia.model.half()

    def generate_audio(self, text, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        # Get audio data and sample rate from inference
        audio_data, sample_rate = self.inference(text, transcript, voice, language, output_file, streaming)
        self.process_audio(audio_data, sample_rate, output_file)

    @torch.no_grad()
    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        transcript = "[S1] "+transcript
        text = ". [S1] "+text + " [S2]"

        return self.dia.generate(
            transcript + text, audio_prompt=os.path.abspath(os.path.join(get_app_root(), str(voice))),
            use_torch_compile=False,
            verbose=True,
            temperature=cfg.get(cfg.dia_temperature) / 100.0,
            top_p=cfg.get(cfg.dia_top_p) / 100.0,
            cfg_filter_top_k = cfg.get(cfg.dia_top_k),
        ), DEFAULT_SAMPLE_RATE