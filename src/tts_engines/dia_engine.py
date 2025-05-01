import os
import sys

import soundfile as sf
import torch

from src.falltalk.config import cfg
from src.tts_engines.tts_engine import tts_engine
from src.falltalk import falltalkutils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dia/dia')))

from dia.dia.model import Dia

class DIA_Engine(tts_engine):

    def __init__(self):
        super().__init__()
        print("Setting Up DIA Engine")
        self.engine_name = 'DIA'
        self.device = cfg.get(cfg.device)
        self.dia = None

    def clean(self):
        self.unload_model()
        self.dia = None

    def unload_model(self):
        self.dia.dac_model.to('cpu')
        self.dia.model.to('cpu')
        super().basic_unload_model()

    def load_base_model(self):
        self.load_model()

    def load_model(self):
        falltalkutils.logger.debug(f"Loading {self.model_path}")


        if self.is_base:
            self.dia = Dia.from_local(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'DIA', 'config.json')), os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'DIA', 'dia-v0_1.pth')), compute_dtype="float16", device=self.device)
        else:
            self.dia = Dia.from_local(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'DIA', 'config.json')), os.path.abspath(self.model_path), compute_dtype="float16", device=self.device)

        self.model = self.dia.model
        self.dia.model = self.dia.model.half()



    def generate_audio(self, text, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        self.inference(text, transcript, voice, language, output_file, streaming)
        rvc_enabled = cfg.get(cfg.rvc_enabled)
        if rvc_enabled and self.rvc_model:
            self.run_rvc(output_file)

        rs_data = falltalkutils.load_audio(output_file, 44100)
        sf.write(output_file, rs_data, 44100, subtype='PCM_16')

    @torch.no_grad()
    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        transcript = "[S1] "+transcript
        text = ". "+text

        output = self.dia.generate(
            transcript + text, audio_prompt=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', str(voice))), use_torch_compile=False, verbose=True
        )
        self.dia.save_audio(output_file, output)

