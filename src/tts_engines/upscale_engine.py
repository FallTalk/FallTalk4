import glob
import os
import sys

import noisereduce as nr
import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model
from pydub import AudioSegment
from pydub.silence import detect_silence, split_on_silence

from src.utils import logging_utils
from src.utils.filesystem_utils import get_app_code_root
from src.utils.file_utils import find_files

sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'audio_upscaler')))
from third_party.audio_upscaler.predict import Predictor
from src.config.config import cfg
from src.utils.audio_utils import load_audio, create_xwm, create_lip_and_fuz, extract_fuz

class UpscaleEngine:

    def __init__(self, parent):
        self.parent = parent
        self.mode = None
        self.demucs_model = None
        self.p = None
        self.sr = 44100
        self.denoise = True

    def upscale_dir(self, parent, directory, mode="denoise", sr=44100, include_subdir=False, replace=True, ddim_steps=50, guidance_scale=3.5):
        """Enhanced upscale_dir that accepts AppCallbacks parent for progress updates."""
        try:
            self.sr = sr
            print(f'Starting Enhancement {mode}')
            self.mode = mode
            if self.mode == 'denoise':
                self.denoise = True

            if self.mode == 'upscale':
                self.p = Predictor()
                self.p.setup(device=cfg.get(cfg.device))

            if self.mode == 'isolate':
                self.demucs_model = get_model('htdemucs_ft')
                self.demucs_model.to(cfg.get(cfg.device))

            print('Getting Files')


            flac_files = find_files('flac', include_subdir, directory)
            wav_files = find_files('wav', include_subdir, directory)
            fuz_files = find_files('fuz', include_subdir, directory)
            xwm_files = find_files('xwm', include_subdir, directory)
            mp3_files = find_files('mp3', include_subdir, directory)

            total = len(wav_files) + len(fuz_files) + len(xwm_files) + len(flac_files) + len(mp3_files)
            count = 0

            parent.on_progress(f"Starting: {count}/{total}")

            for flac_file in flac_files:
                self.do_flac(ddim_steps, guidance_scale,  replace, flac_file)
                count += 1
                parent.on_progress(f"Enhancement: {count}/{total}")

            for mp3_file in mp3_files:
                self.do_mp3(ddim_steps, guidance_scale,  replace, mp3_file)
                count += 1
                parent.on_progress(f"Enhancement: {count}/{total}")

            for wav_file in wav_files:
                self.do_wav(ddim_steps, guidance_scale,  replace, wav_file)
                count += 1
                parent.on_progress(f"Enhancement: {count}/{total}")

            for xwm_file in xwm_files:
                self.do_xwm(ddim_steps, guidance_scale,  replace, xwm_file)
                count += 1
                parent.on_progress(f"Enhancement: {count}/{total}")

            for fuz_file in fuz_files:
                self.do_fuz(ddim_steps, guidance_scale,  replace, fuz_file)
                count += 1
                parent.on_progress(f"Enhancement: {count}/{total}")

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

            parent.on_done()
        except Exception as e:
            logging_utils.logger.exception(f"Error: {e}")
            parent.on_error("Error During Enhancement", "An Error Occured while loading the cleaner. Please check your logs and report the issue if needed")

    def do_mp3(self, ddim_steps, guidance_scale,  replace, mp3_file):
        try:
            audio = AudioSegment.from_mp3(mp3_file)
            wav_file = mp3_file.replace(".mp3", ".wav")
            audio.export(wav_file, format="wav")

            self.do_upscale(wav_file, wav_file, ddim_steps, guidance_scale)

            if replace:
                output_file = mp3_file
            else:
                output_file = mp3_file.replace(".mp3", "_enhanced.mp3")

            audio = AudioSegment.from_wav(wav_file)
            audio.export(output_file, format="mp3")

            if os.path.exists(wav_file):
                os.remove(wav_file)

        except Exception as e:
            logging_utils.logger.exception(f"Error: {e}")

    def do_flac(self, ddim_steps, guidance_scale,  replace, flac_file):
        try:
            data, sample_rate = sf.read(flac_file)
            wav_file = flac_file.replace(".flac", ".wav")
            sf.write(wav_file, data, sample_rate)

            self.do_upscale(wav_file, wav_file, ddim_steps, guidance_scale)

            if replace:
                output_file = flac_file
            else:
                output_file = flac_file.replace(".flac", "_enhanced.flac")

            data, sample_rate = sf.read(wav_file)
            sf.write(output_file, data, sample_rate)

            if os.path.exists(wav_file):
                os.remove(wav_file)

        except Exception as e:
            logging_utils.logger.exception(f"Error: {e}")

    def do_wav(self, ddim_steps, guidance_scale,  replace, wav_file):
        try:
            if replace:
                output_file = wav_file
            else:
                output_file = wav_file.replace(".wav", "_enhanced.wav")

            self.do_upscale(wav_file, output_file, ddim_steps, guidance_scale)

        except Exception as e:
            logging_utils.logger.exception(f"Error: {e}")

    def do_xwm(self, ddim_steps, guidance_scale,  replace, xwm_file):
        try:
            wav_file = xwm_file.replace(".xwm", ".wav")
            create_xwm(xwm_file, wav_file, encode=False)

            self.do_upscale(wav_file, wav_file, ddim_steps, guidance_scale)

            if replace:
                create_xwm(wav_file, xwm_file, encode=True)
        except Exception as e:
            logging_utils.logger.exception(f"Error: {e}")

    def do_fuz(self, ddim_steps, guidance_scale,  replace, fuz_file):
        try:
            extract_fuz(fuz_file)
            xwm_file = fuz_file.replace(".fuz", ".xwm")
            wav_file = fuz_file.replace(".fuz", ".wav")
            create_xwm(xwm_file, wav_file, encode=False)

            self.do_upscale(wav_file, wav_file, ddim_steps, guidance_scale)

            if replace:
                create_lip_and_fuz(self.parent, wav_file, True)
        except Exception as e:
            logging_utils.logger.exception(f"Error: {e}")

    def do_upscale(self, input_file, output_file, ddim_steps, guidance_scale):
        if self.p:
            self.p.predict(
                input_file,
                output_file,
                sr=self.sr,
                ddim_steps=ddim_steps,
                guidance_scale=guidance_scale,
                seed=cfg.get(cfg.seed)
            )
            self.remove_silence_at_end(output_file)

        if self.demucs_model:
            self.demucs_file(input_file, output_file)

        if self.denoise:
            self.denoise_file(input_file, output_file)

    def upscale_file(self, input_file, replace=True, sr=44100, ddim_steps=50, guidance_scale=3.5, model_name="basic"):
        self.p = Predictor()
        self.p.setup(model_name, cfg.get(cfg.device))

        if ".wav" in input_file:
            self.do_wav(sr, ddim_steps, guidance_scale,  replace, input_file)

        if ".xwm" in input_file:
            self.do_xwm(sr, ddim_steps, guidance_scale,  replace, input_file)

        if ".fuz" in input_file:
            self.do_fuz(sr, ddim_steps, guidance_scale,  replace, input_file)

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

        # Add 400ms padding to the end
        padding = AudioSegment.silent(duration=400)
        audio = audio + padding

        # Export the modified audio
        audio.export(wav_file, format="wav")

    def denoise_file(self, input_file, output_file):
        data = load_audio(input_file, sampling_rate=self.sr)
        reduced_noise = nr.reduce_noise(y=data, sr=self.sr, device=cfg.get(cfg.device))
        sf.write(output_file, reduced_noise, self.sr)

    def demucs_file(self, input_file, output_file, save_to_mono=True):
        audio = load_audio(input_file, sampling_rate=self.sr, channels=2)

        audio = torch.tensor(audio).float()
        audio = audio.view(1, 2, -1)  # Reshape to (batch, channels, length)

        sources = apply_model(self.demucs_model, torch.tensor(audio).float(), device=cfg.get(cfg.device), num_workers=os.cpu_count())

        # Extract the vocals from the sources (assuming vocals are the fourth source)
        vocals = sources[:, 3, :, :]  # Shape: (batch, channels, length)

        # Normalize the vocals
        max_val = torch.max(torch.abs(vocals))
        if max_val > 0:
            vocals /= max_val

        # Convert the vocals to a numpy array
        vocals_np = vocals.squeeze(0).cpu().numpy()

        # If the input was mono, convert the output to mono
        if save_to_mono:
            vocals_np = np.mean(vocals_np, axis=0)  # Average the two channels to create mono

        # Transpose the array to match the expected shape for writing
        vocals_np = vocals_np.T

        # Write the vocals to the output file
        sf.write(output_file, vocals_np, self.demucs_model.samplerate)

    def split_wav(self, input_file, min_silence_len=1000, silence_thresh=-40, keep_silence=500):
        data = AudioSegment.from_wav(input_file)

        # Split the audio by silence
        chunks = split_on_silence(
            data,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )

        output_folder = os.path.dirname(input_file)
        base_name = os.path.basename(input_file)
        file_name, file_ext = os.path.splitext(base_name)

        for i, chunk in enumerate(chunks):
            output_file = f"{output_folder}/{file_name}_chunk{i + 1}{file_ext}"
            chunk.export(output_file, format="wav")
