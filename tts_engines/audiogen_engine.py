import PySide6
import torch
import torchaudio
from PySide6.QtCore import QMetaObject, Q_ARG, Qt
from audiocraft.data.audio import audio_write
from audiocraft.models import musicgen, AudioGen

import falltalkutils
from falltalk.config import cfg


class AudioGenEngine:

    def __init__(self, parent):
        self.parent = parent
        self.model = None

    def clean(self):
        del self.model
        self.model = None
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

            self.model.set_generation_params(duration=float(audio_length_in_s))

            data = self.model.generate([prompt])

            audio_write(f'{output_file}', data[0].cpu(), self.model.sample_rate, strategy="loudness")

            QMetaObject.invokeMethod(self.parent, "updateMediaplayer", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, panel), Q_ARG(str, f'{output_file}.wav'))
            QMetaObject.invokeMethod(self.parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, self.parent))
        except Exception as e:
            falltalkutils.logger.exception(f"Error: {e}")
            QMetaObject.invokeMethod(self.parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, self.parent), Q_ARG(str, "Error During Generation"), Q_ARG(str, "An Error Occured while generating sounds. Please check your logs and report the issue if needed"))
