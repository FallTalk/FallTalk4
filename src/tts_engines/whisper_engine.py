import re
from typing import Union

import numpy as np
from num2words import num2words
import librosa
from src.config.config import cfg

from whisperx import load_model

class WhisperxAlignModel:
    def __init__(self):
        from whisperx import load_align_model
        self.model, self.metadata = load_align_model(language_code="en", device=cfg.get(cfg.device))

    def align(self, segments, audio_path: Union[str, np.ndarray]):
        from whisperx import align, load_audio
        if isinstance(audio_path, str):
            audio = load_audio(audio_path)
        else:
            audio = audio_path
            # Fix dimensionality issue: if audio has shape (1, data), reshape to (data,)
            if isinstance(audio, np.ndarray) and audio.ndim > 1 and audio.shape[0] == 1:
                audio = audio.flatten()
        return align(segments, self.model, self.metadata, audio, cfg.get(cfg.device), return_char_alignments=False)["segments"]


class WhisperxModel:
    def __init__(self, model_name, align_model: WhisperxAlignModel):
        self.device = cfg.get(cfg.device)
        if self.device == "cpu":
            self.compute_type = "float32"
        else:
            self.compute_type = "float16"
        self.align_model = align_model
        self.model_name = model_name
        self.model = None

    def transcribe(self, audio: Union[str, np.ndarray]):
        if self.model is None:
            self.model = load_model(self.model_name, self.device, language='en', compute_type=self.compute_type, asr_options={"suppress_numerals": True, "max_new_tokens": None, "clip_timestamps": None, "hallucination_silence_threshold": None})
        self.model.model.model.load_model(True)

        # Fix dimensionality issue: if audio has shape (1, data), reshape to (data,)
        if isinstance(audio, np.ndarray) and audio.ndim > 1 and audio.shape[0] == 1:
            audio = audio.flatten()

        segments = self.model.transcribe(audio, batch_size=8)["segments"]
        for segment in segments:
            segment['text'] = replace_numbers_with_words(segment['text'])

        #VRAM Savings
        self.model.model.model.unload_model(True)
        return self.align_model.align(segments, audio)


def replace_numbers_with_words(sentence):
    sentence = re.sub(r'(\d+)', r' \1 ', sentence)  # add spaces around numbers

    def replace_with_words(match):
        num = match.group(0)
        try:
            return num2words(num)  # Convert numbers to words
        except:
            return num  # In case num2words fails (unlikely with digits but just to be safe)

    return re.sub(r'\b\d+\b', replace_with_words, sentence)  # Regular expression that matches numbers


def get_transcribe_state(segments):
    words_info = [word_info for segment in segments for word_info in segment["words"]]
    transcript = " ".join([segment["text"] for segment in segments])
    if transcript is not None and len(transcript) > 0:
        transcript = transcript[1:] if transcript[0] == " " else transcript
        return {
            "segments": segments,
            "transcript": transcript,
            "words_info": words_info,
            "transcript_with_start_time": " ".join([f"{word['start']} {word['word']}" for word in words_info]),
            "transcript_with_end_time": " ".join([f"{word['word']} {word['end']}" for word in words_info]),
            "word_bounds": [f"{word['start']} {word['word']} {word['end']}" for word in words_info]
        }
    else:
        return None

class Whisper_Engine():
    def __init__(self):
        self.align_model = WhisperxAlignModel()
        self.transcribe_model = WhisperxModel("distil-large-v3.5", self.align_model)

    def transcribe(self, audio: Union[str, np.ndarray], sr: float = None):
        if isinstance(audio, np.ndarray) and audio.ndim > 1 and audio.shape[0] == 1:
            audio = audio.flatten()

        if isinstance(audio, np.ndarray) and sr is not None and sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        segments = self.transcribe_model.transcribe(audio)
        return get_transcribe_state(segments)

    def clean(self):
        del self.transcribe_model.model
        del self.align_model
        del self.transcribe_model
