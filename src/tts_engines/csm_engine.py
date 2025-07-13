import hashlib
import os

import librosa
import numpy as np
import torch
from transformers import AutoProcessor, CsmForConditionalGeneration

from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.tts_engines.tts_engine import tts_engine
from src.utils import torch_utils
from src.utils.audio_utils import load_audio
from src.utils.filesystem_utils import get_app_root

def string_to_int(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (10 ** 8)


class CSMEngine(tts_engine):

    def __init__(self):
        super().__init__()
        print("Setting Up CSM Engine")
        self.engin_type = EngineType.CSM
        self.engine_name = self.engin_type.value
        self.device = cfg.get(cfg.device)
        self.processor = None
        self.tokenize = None

    def generate_audio(self, text, transcript=None, voice=None, language='en', output_file=None, streaming=False, speaker=None):
        # Get audio data and sample rate from inference
        audio_data, sample_rate = self.inference(text, transcript, voice, language, output_file, streaming, speaker)
        self.process_audio(audio_data, sample_rate, output_file)

    def load_model(self):
        print("Loading CSM Model")

        if self.is_base:
            self.processor = AutoProcessor.from_pretrained(
                str(os.path.abspath(os.path.join(get_app_root(), 'models', 'CSM', '1b'))), torch_dtype=torch_utils.get_compute_dtype())
            self.model = CsmForConditionalGeneration.from_pretrained(
                str(os.path.abspath(os.path.join(get_app_root(), 'models', 'CSM', '1b'))), device_map=self.device, torch_dtype=torch_utils.get_compute_dtype())
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_path, torch_dtype=torch_utils.get_compute_dtype())
            self.model = CsmForConditionalGeneration.from_pretrained(self.model_path, device_map=self.device, torch_dtype=torch_utils.get_compute_dtype())

        if self.device == 'cpu':
            self.model.eval().cpu()
        else:
            self.model.eval()

    def unload_model(self):
        self.basic_unload_model()
        del self.processor
        self.processor = None

    def load_audio_t(self, voice, sr):
        audio, _ = librosa.load(voice, sr=sr)
        return audio

    def trim_trailing_silence(self, audio, threshold=0.01):
        # Find where audio drops below threshold
        trailing_silence = np.where(np.abs(audio[::-1]) > threshold)[0]
        if len(trailing_silence) > 0:
            return audio[:-trailing_silence[0]]
        return audio

    @torch.no_grad()
    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False,
                  speaker=None):
        speaker_id = "0" if speaker is None else string_to_int(speaker)
        conversation = []

        if voice is not None and transcript is not None:
            conversation.append({
                "role": f"{speaker_id}",
                "content": [{"type": "text", "text": transcript}, {"type": "audio", "path": self.load_audio_t(voice, 24000)}],
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
        ).to(self.device).to(torch_utils.get_compute_dtype())

        # infer the model
        audio_values = self.model.generate(**inputs, output_audio=True)
        audio, sr = audio_values[0].cpu().float().numpy(), 24000

        # Remove that annoying pop at the end
        samples_to_silence = int(0.1 * sr)  # 0.2s * 24000 samples/s = 4800 samples

        processed_audio = audio.copy()

        if len(audio) > samples_to_silence:
            processed_audio[-samples_to_silence:] = 0.0

        trimmed_audio = processed_audio

        return trimmed_audio, sr
