from src.config.config import cfg

import sys
import os
import torch
from src.tts_engines.tts_engine import tts_engine
from src.utils.file_utils import get_app_root

sys.path.append(os.path.abspath(os.path.join(get_app_root(), 'orpheus', 'orpheus_tts_pypi', 'orpheus_tts')))

class OrpheusEngine(tts_engine):

    def __init__(self):
        super().__init__()
        print("Setting Up Orpheus Engine")
        self.engine_name = 'Orpheus'
        self.device = cfg.get(cfg.device)
        self.fish = None

    def generate_audio(self, text, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        self.inference(text, transcript, voice, language, output_file, streaming)

    @torch.no_grad()
    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        print("Loading Orpheus Model")