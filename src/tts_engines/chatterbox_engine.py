import os
import sys

import torch

from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.tts_engines.tts_engine import tts_engine
from src.utils import logging_utils
from src.utils.filesystem_utils import get_app_root, get_app_code_root

sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'chatterbox', 'src')))

from third_party.chatterbox.src.chatterbox import ChatterboxTTS


class ChatterboxEngine(tts_engine):

    def __init__(self):
        super().__init__()
        print("Setting Up Chatterbox Engine")
        self.engin_type = EngineType.CHATTERBOX
        self.engine_name = self.engin_type.value
        self.device = cfg.get(cfg.device)

    def clean(self):
        self.unload_model()

    def unload_model(self):
        super().basic_unload_model()


    def load_model(self):
        logging_utils.logger.debug(f"Loading {self.model_path}")
        if self.is_base:
            self.model = ChatterboxTTS.from_local(ckpt_dir = os.path.abspath(os.path.join(get_app_root(), 'models', 'Chatterbox', '0.5B')), device=self.device)
        else:
            self.model = ChatterboxTTS.from_local(self.model_path, device=self.device)

    def generate_audio(self, text, transcript=None, voice=None, language='en', output_file=None, streaming=False, speaker=None):
        # Get audio data and sample rate from inference
        audio_data, sample_rate = self.inference(text, transcript, voice, language, output_file, streaming)
        self.process_audio(audio_data, sample_rate, output_file)

    @torch.no_grad()
    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        wav = self.model.generate(text,
                                  audio_prompt_path=voice,
                                  repetition_penalty=cfg.get(cfg.chatterbox_repetition_penalty) / 10.0,
                                  top_p=cfg.get(cfg.chatterbox_top_p) / 100.0,
                                  min_p=cfg.get(cfg.chatterbox_min_p) / 1000.0,
                                  exaggeration=cfg.get(cfg.chatterbox_exaggeration) / 100.0,
                                  cfg_weight=cfg.get(cfg.chatterbox_cfg_weight) / 100.0
                                  )
        return wav.cpu().float().numpy(), self.model.sr
