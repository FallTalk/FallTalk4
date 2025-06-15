import librosa
import numpy as np

from enums.engine_type import EngineType
from src.tts_engines.tts_engine import tts_engine
import soundfile as sf
from src.utils.audio_utils import load_audio

class RVC_Engine(tts_engine):
    def __init__(self):
        super().__init__()
        self.engin_type = EngineType.RVC
        self.engine_name = self.engin_type.value

    def load_model(self):
        pass

    def load_base_model(self):
        pass

    def clean(self):
        super().clean()

    def unload_model(self):
        pass

    def run_rvc_file(self, input_tts_path):
        audio_data, sample_rate = super().run_rvc_file(input_tts_path)

        if not np.issubdtype(audio_data.dtype, np.floating):
            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

        if sample_rate != 44100:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=44100)
            sample_rate = 44100

        sf.write(input_tts_path, audio_data, sample_rate, subtype='PCM_16')
