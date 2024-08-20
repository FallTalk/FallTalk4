import PySide6
import soundfile as sf
import torch
from PySide6.QtCore import QMetaObject, Q_ARG, Qt
from diffusers import AudioLDM2Pipeline

import falltalkutils
from falltalk.config import cfg
from diffusers import DPMSolverMultistepScheduler


class Audioldm2Engine:

    def __init__(self, parent, repo_id):
        self.parent = parent
        self.repo_id = repo_id
        self.pipe = AudioLDM2Pipeline.from_pretrained(self.repo_id, torch_dtype=torch.float16)
        #self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        if cfg.get(cfg.engine) != "cpu":
            self.pipe.enable_model_cpu_offload()
            self.generator = torch.Generator("cuda").manual_seed(0)
        else:
            self.generator = torch.Generator("cpu").manual_seed(0)
            self.pipe.to("cpu")

    def progress_callback(self, current_step: int, total_steps: int, tensor: torch.Tensor):
        falltalkutils.logger.debug(f"Audio Generation {current_step}/{total_steps}")
        percentage = ((current_step + 1) / total_steps) * 100
        if percentage > 100:
            percentage = 100
        QMetaObject.invokeMethod(self.parent, "update_loader", Qt.QueuedConnection, Q_ARG(str, f"Estimated Progress: {percentage:.2f}%"))

    def clean(self):
        del self.pipe
        self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, prompt, output_file, panel, audio_length_in_s=10.0, negative_prompt="Low quality, average quality."):
        try:
            print(f"Starting gen for prompt {prompt}")
            data = self.pipe(prompt, num_inference_steps=200, negative_prompt=negative_prompt, audio_length_in_s=audio_length_in_s, num_waveforms_per_prompt=3, generator=self.generator, callback=self.progress_callback).audios[0]
            sf.write(output_file, data, 16000)

            QMetaObject.invokeMethod(self.parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, self.parent))
            QMetaObject.invokeMethod(self.parent, "updateMediaplayer", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, panel), Q_ARG(str, output_file))
        except Exception as e:
            falltalkutils.logger.exception(f"Error: {e}")
            QMetaObject.invokeMethod(self.parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, self.parent), Q_ARG(str, "Error During Generation"), Q_ARG(str, "An Error Occured while generating sounds. Please check your logs and report the issue if needed"))
