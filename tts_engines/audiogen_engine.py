import os

import PySide6
import torch
import torchaudio
from PySide6.QtCore import QMetaObject, Q_ARG, Qt
from audiocraft.data.audio import audio_write
from audiocraft.models import musicgen, AudioGen
from pydub import AudioSegment

import falltalkutils
from falltalk.config import cfg


class AudioGenEngine:

    def __init__(self, parent):
        self.parent = parent
        self.model = None
        self.processor = None

    def clean(self):
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def progress_callback(self, current_step: int, total_steps: int):
        falltalkutils.logger.debug(f"Audio Generation {current_step}/{total_steps}")
        percentage = ((current_step + 1) / total_steps) * 100
        if percentage > 100:
            percentage = 100
        QMetaObject.invokeMethod(self.parent, "update_loader", Qt.QueuedConnection, Q_ARG(str, f"Estimated Progress: {percentage:.2f}%"))

    def generate(self, prompt, output_file, panel, audio_length_in_s=10.0):
        try:
            if self.model is None:
                self.model = AudioGen.get_pretrained('facebook/audiogen-medium', device=cfg.get(cfg.device))
                self.model.set_custom_progress_callback(self.progress_callback)

            if cfg.get(cfg.parse_mode) == 'split':
                prompts = [item.strip() for item in prompt.split(',')]
            else:
                prompts = [prompt.strip()]
            self.model.set_generation_params(duration=float(audio_length_in_s), temperature=float(cfg.get(cfg.music_temperature) / 100.0))
            data = self.model.generate(prompts, progress=True)
            wav_files = []
            for idx, one_wav in enumerate(data):
                # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
                wav_files.append(audio_write(f'temp/{idx}', one_wav.detach().cpu(), self.model.sample_rate, strategy="loudness", loudness_compressor=True))

            layered = None

            for wav_file in wav_files:
                audio = AudioSegment.from_wav(wav_file)
                if layered is None:
                    layered = audio
                else:
                    layered = layered.overlay(audio)
                os.remove(wav_file)

            if layered is not None:
                layered.export(f'{output_file}.wav', format="wav")

            del data
            del wav_files

            QMetaObject.invokeMethod(self.parent, "updateMediaplayer", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, panel), Q_ARG(str, f'{output_file}.wav'))
            QMetaObject.invokeMethod(self.parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, self.parent))
        except Exception as e:
            falltalkutils.logger.exception(f"Error: {e}")
            QMetaObject.invokeMethod(self.parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, self.parent), Q_ARG(str, "Error During Generation"), Q_ARG(str, "An Error Occured while generating sounds. Please check your logs and report the issue if needed"))
