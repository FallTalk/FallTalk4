import glob
import os

import PySide6
import numpy as np
import torch
from PySide6.QtCore import QMetaObject, Qt, Q_ARG
from pydub import AudioSegment
from pydub.silence import detect_silence

import falltalkutils
from audio_upscaler.predict import Predictor
from falltalk.config import cfg
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.audio import AudioFile

import soundfile as sf


class UpscaleEngine:

    def __init__(self, parent):
        self.parent = parent
        self.mode = None
        self.demucs_model = None
        self.p = None
        self.sr = 44100

    def upscale_dir(self, directory, replace=True, include_subdir=False, mode="denoise", sr=44100, ddim_steps=50, guidance_scale=3.5, seed=None):
        try:
            self.sr = sr
            print(f'Starting Enhancement {mode}')
            self.mode = mode
            if self.mode == 'denoise' or self.mode == 'both':
                self.demucs_model = get_model('htdemucs_ft')
                self.demucs_model.to(cfg.get(cfg.device))

            if self.mode == 'upscale' or self.mode == 'both':
                self.p = Predictor()
                self.p.setup(device=cfg.get(cfg.device))

            print('Getting Files')

            wav_files = glob.glob(os.path.join(directory, '**', '*.wav'), recursive=include_subdir)
            fuz_files = glob.glob(os.path.join(directory, '**', '*.fuz'), recursive=include_subdir)
            xwm_files = glob.glob(os.path.join(directory, '**', '*.xwm'), recursive=include_subdir)

            total = len(wav_files) + len(fuz_files) + len(xwm_files)
            count = 0

            QMetaObject.invokeMethod(self.parent, "update_loader", Qt.QueuedConnection, Q_ARG(str, f"Starting: {count}/{total}"))

            for wav_file in wav_files:
                self.do_wav(ddim_steps, guidance_scale, seed, replace, wav_file)
                count += 1
                QMetaObject.invokeMethod(self.parent, "update_loader", Qt.QueuedConnection, Q_ARG(str, f"Enhancement: {count}/{total}"))

            for xwm_file in xwm_files:
                self.do_xwm(ddim_steps, guidance_scale, seed, replace, xwm_file)
                count += 1
                QMetaObject.invokeMethod(self.parent, "update_loader", Qt.QueuedConnection, Q_ARG(str, f"Enhancement: {count}/{total}"))

            for fuz_file in fuz_files:
                self.do_fuz(ddim_steps, guidance_scale, seed, replace, fuz_file)
                count += 1
                QMetaObject.invokeMethod(self.parent, "update_loader", Qt.QueuedConnection, Q_ARG(str, f"Enhancement: {count}/{total}"))

            if self.p:
                self.p.audiosr.to('cpu')
                del self.p.audiosr
                del self.p
                self.p = None

            if self.demucs_model:
                self.demucs_model.to('cpu')
                del self.demucs_model
                self.demucs_model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            QMetaObject.invokeMethod(self.parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, self.parent))
        except Exception as e:
            falltalkutils.logger.exception(f"Error: {e}")
            QMetaObject.invokeMethod(self.parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, self.parent), Q_ARG(str, "Error During Enhancement"), Q_ARG(str, "An Error Occured while loading the cleaner. Please check your logs and report the issue if needed"))

    def do_wav(self, ddim_steps, guidance_scale, seed, replace, wav_file):
        try:
            if replace:
                output_file = wav_file
            else:
                output_file = wav_file.replace(".wav", "_enhanced.wav")

            if self.p:
                self.p.predict(
                    wav_file,
                    output_file,
                    sr=self.sr,
                    ddim_steps=ddim_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
                self.remove_silence_at_end(output_file)

            if self.demucs_model:
                self.demucs_file(wav_file if self.mode != 'both' else output_file, output_file)

        except Exception as e:
            falltalkutils.logger.exception(f"Error: {e}")

    def do_xwm(self, ddim_steps, guidance_scale, seed, replace, xwm_file):
        try:
            wav_file = xwm_file.replace(".xwm", ".wav")
            falltalkutils.create_xwm(xwm_file, wav_file, encode=False)

            if self.p:
                self.p.predict(
                    wav_file,
                    wav_file,
                    sr=self.sr,
                    ddim_steps=ddim_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
                self.remove_silence_at_end(wav_file)

            if self.demucs_model:
                self.demucs_file(wav_file, wav_file)

            if replace:
                falltalkutils.create_xwm(wav_file, xwm_file, encode=True)
        except Exception as e:
            falltalkutils.logger.exception(f"Error: {e}")

    def do_fuz(self, ddim_steps, guidance_scale, seed, replace, fuz_file):
        try:
            falltalkutils.extract_fuz(fuz_file)
            xwm_file = fuz_file.replace(".fuz", ".xwm")
            wav_file = fuz_file.replace(".fuz", ".wav")
            falltalkutils.create_xwm(xwm_file, wav_file, encode=False)

            if self.p:
                self.p.p.predict(
                    wav_file,
                    wav_file,
                    sr=self.sr,
                    ddim_steps=ddim_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )

                self.remove_silence_at_end(wav_file)

            if self.demucs_model:
                self.demucs_file(wav_file, wav_file)

            if replace:
                falltalkutils.create_lip_and_fuz(self.parent, wav_file, True)
        except Exception as e:
            falltalkutils.logger.exception(f"Error: {e}")

    def upscale_file(self, input_file, replace=True, sr=44100, ddim_steps=50, guidance_scale=3.5, model_name="basic", seed=None):
        self.p = Predictor()
        self.p.setup(model_name, cfg.get(cfg.device))

        if ".wav" in input_file:
            self.do_wav(sr, ddim_steps, guidance_scale, seed, replace, input_file)

        if ".xwm" in input_file:
            self.do_xwm(sr, ddim_steps, guidance_scale, seed, replace, input_file)

        if ".fuz" in input_file:
            self.do_fuz(sr, ddim_steps, guidance_scale, seed, replace, input_file)

        self.p.audiosr.to('cpu')
        del self.p.audiosr
        del self.p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def remove_silence_at_end(self, wav_file, silence_threshold=-60, min_silence_len=500):
        # Load the audio file
        audio = AudioSegment.from_wav(wav_file)

        # Detect silence in the audio
        silence_ranges = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_threshold)

        # Upscaling min length is 5 seconds, so anything scaled will be moved to 5 sec, even if shorter. Detect and remove the silence.
        if len(audio) > 6000:
            print(f"Skipping {wav_file} trim as it is longer than 6 seconds.")
            return

        # If there is silence at the end, remove it
        if silence_ranges:
            last_silence = silence_ranges[-1]
            if last_silence[1] == len(audio):
                audio = audio[:last_silence[0]]

        # Export the modified audio
        output_file = wav_file.replace('.wav', '_trimmed.wav')
        audio.export(output_file, format="wav")
        print(f"Trimmed audio saved as {output_file}")

    def demucs_file(self, input_file, output_file):
        # Read the audio file with the correct samplerate
        audio = AudioFile(input_file).read(streams=0, samplerate=self.sr)

        # Check if the input audio is mono
        is_mono = audio.shape[0] == 1

        # Ensure the audio tensor has 2 channels (stereo)
        if is_mono:
            audio = torch.cat([audio, audio], dim=0)  # Duplicate the mono channel to create stereo

        # Ensure the audio tensor has the correct shape (batch, channels, length)
        if len(audio.shape) == 2:
            audio = audio.unsqueeze(0)  # Add batch dimension

        # Apply the model to separate sources
        sources = apply_model(self.demucs_model, audio.float(), device=cfg.get(cfg.device), num_workers=os.cpu_count())

        # Extract the vocals from the sources (assuming vocals are the fourth source)
        vocals = sources[:, 3, :, :]  # Shape: (batch, channels, length)

        # Normalize the vocals
        max_val = torch.max(torch.abs(vocals))
        if max_val > 0:
            vocals /= max_val

        # Convert the vocals to a numpy array
        vocals_np = vocals.squeeze(0).cpu().numpy()

        # If the input was mono, convert the output to mono
        if is_mono:
            vocals_np = np.mean(vocals_np, axis=0)  # Average the two channels to create mono

        # Transpose the array to match the expected shape for writing
        vocals_np = vocals_np.T

        # Write the vocals to the output file
        sf.write(output_file, vocals_np, self.demucs_model.samplerate)
