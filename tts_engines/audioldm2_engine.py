import PySide6
import soundfile as sf
import torch
from PySide6.QtCore import QMetaObject, Q_ARG, Qt
from diffusers import AudioLDM2Pipeline

import falltalkutils


class Audioldm2Engine:

    def __init__(self, parent, repo_id):
        self.parent = parent
        self.repo_id = repo_id
        self.pipe = AudioLDM2Pipeline.from_pretrained(self.repo_id, torch_dtype=torch.float16)
        self.pipe.enable_model_cpu_offload()


    def generate(self, prompt, output_file, panel, negative_prompt="Low quality.", audio_length_in_s=10.0):
        try:
            falltalkutils.seed_everything(0)
            data = self.pipe(prompt, num_inference_steps=200, negative_prompt=negative_prompt, audio_length_in_s=audio_length_in_s, num_waveforms_per_prompt=3).audios[0]
            sf.write(output_file, data, 16000)

            QMetaObject.invokeMethod(self.parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, self.parent))
            QMetaObject.invokeMethod(self.parent, "updateMediaplayer", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, panel), Q_ARG(str, output_file))
        except Exception as e:
            falltalkutils.logger.exception(f"Error: {e}")
            QMetaObject.invokeMethod(self.parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, self.parent), Q_ARG(str, "Error During Generation"), Q_ARG(str, "An Error Occured while generating sounds. Please check your logs and report the issue if needed"))

