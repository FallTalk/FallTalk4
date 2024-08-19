import glob
import os

import PySide6
import torch
from PySide6.QtCore import QMetaObject, Qt, Q_ARG

import falltalkutils
from audio_upscaler.predict import Predictor
from falltalk.config import cfg


class UpscaleEngine:

    def __init__(self, parent):
        self.parent = parent

    def upscale_dir(self, directory, replace=True, include_subdir=False, model_name="speech", sr=44100, ddim_steps=50, guidance_scale=3.5, seed=None):
        try:
            p = Predictor()
            p.setup(model_name, cfg.get(cfg.device))

            wav_files = glob.glob(os.path.join(directory, '**', '*.wav'), recursive=include_subdir)
            fuz_files = glob.glob(os.path.join(directory, '**', '*.fuz'), recursive=include_subdir)
            xwm_files = glob.glob(os.path.join(directory, '**', '*.xwm'), recursive=include_subdir)

            total = len(wav_files)+len(fuz_files)+len(xwm_files)
            count = 0

            QMetaObject.invokeMethod(self.parent, "update_loader", Qt.QueuedConnection, Q_ARG(str, f"Starting...: {count}/{total}"))

            for wav_file in wav_files:
                self.do_wav(p, sr, ddim_steps, guidance_scale, seed, replace, wav_file)
                count += 1
                QMetaObject.invokeMethod(self.parent, "update_loader", Qt.QueuedConnection, Q_ARG(str, f"Upscaling...: {count}/{total}"))

            for xwm_file in xwm_files:
                self.do_xwm(p, sr, ddim_steps, guidance_scale, seed, replace, xwm_file)
                count += 1
                QMetaObject.invokeMethod(self.parent, "update_loader", Qt.QueuedConnection, Q_ARG(str, f"Upscaling...: {count}/{total}"))

            for fuz_file in fuz_files:
                self.do_fuz(p, sr, ddim_steps, guidance_scale, seed, replace, fuz_file)
                count += 1
                QMetaObject.invokeMethod(self.parent, "update_loader", Qt.QueuedConnection, Q_ARG(str, f"Upscaling...: {count}/{total}"))

            del p.audiosr
            del p
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            QMetaObject.invokeMethod(self.parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, self.parent))
        except Exception as e:
            falltalkutils.logger.exception(f"Error: {e}")
            QMetaObject.invokeMethod(self.parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, self.parent), Q_ARG(str, "Error During Upscale"), Q_ARG(str, "An Error Occured while loading the upscaler. Please check your logs and report the issue if needed"))

    def do_wav(self, p, sr, ddim_steps, guidance_scale, seed, replace, wav_file):
        if replace:
            output_file = wav_file
        else:
            output_file = wav_file.replace(".wav", "_upscale.wav")

        p.predict(
            wav_file,
            output_file,
            sr=sr,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )

    def do_xwm(self, p, sr, ddim_steps, guidance_scale, seed, replace, xwm_file):
        wav_file = xwm_file.replace(".xwm", ".wav")
        falltalkutils.create_xwm(xwm_file, wav_file, encode=False)

        p.predict(
            wav_file,
            wav_file,
            sr=sr,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )

        if replace:
            falltalkutils.create_xwm(wav_file, xwm_file, encode=True)

    def do_fuz(self, p, sr, ddim_steps, guidance_scale, seed, replace, fuz_file):
        falltalkutils.extract_fuz(fuz_file)
        xwm_file = fuz_file.replace(".fuz", ".xwm")
        wav_file = fuz_file.replace(".fuz", ".wav")
        falltalkutils.create_xwm(xwm_file, wav_file, encode=False)

        p.predict(
            wav_file,
            wav_file,
            sr=sr,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )

        if replace:
            falltalkutils.create_lip_and_fuz(self.parent, wav_file)

    def upscale_file(self, input_file, replace=True, sr=44100, ddim_steps=50, guidance_scale=3.5, model_name="basic", seed=None):
        p = Predictor()
        p.setup(model_name, cfg.get(cfg.device))

        if ".wav" in input_file:
            self.do_wav(p, sr, ddim_steps, guidance_scale, seed, replace, input_file)

        if ".xwm" in input_file:
            self.do_xwm(p, sr, ddim_steps, guidance_scale, seed, replace, input_file)

        if ".fuz" in input_file:
            self.do_fuz(p, sr, ddim_steps, guidance_scale, seed, replace, input_file)

        del p.audiosr
        del p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
