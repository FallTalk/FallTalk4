import falltalkutils
from tts_engine import tts_engine
import soundfile as sf


class RVC_Engine(tts_engine):
    def __init__(self):
        super().__init__()
        self.engine_name = 'RVC'

    def load_model(self):
        pass

    def load_base_model(self):
        pass

    def clean(self):
        super().clean()

    def unload_model(self):
        pass

    def run_rvc(self, input_tts_path):
        super().run_rvc(input_tts_path)

        rs_data = falltalkutils.load_audio(input_tts_path, 44100)
        sf.write(input_tts_path, rs_data, 44100, subtype='PCM_16')
