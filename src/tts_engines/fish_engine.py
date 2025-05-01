import src.falltalk.falltalkutils
from src.falltalk.config import cfg

import sys
import os
import torch
from src.tts_engines.tts_engine import tts_engine

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fish-speech/fish_speech')))

class FishSpeechEngine(tts_engine):

    def __init__(self):
        super().__init__()
        print("Setting Up FishSpeech Engine")
        self.engine_name = 'FishSpeech'
        self.device = cfg.get(cfg.device)
        self.fish = None

    def generate_audio(self, text, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        self.inference(text, transcript, voice, language, output_file, streaming)

    @torch.no_grad()
    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        print("Loading Fish Model")

