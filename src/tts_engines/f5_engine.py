import math
import os
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from hydra.utils import get_class
from omegaconf import OmegaConf

from src.config.config import cfg
from src.tts_engines.tts_engine import tts_engine
from src.utils.file_utils import get_app_root
from src.utils import logging_utils
from src.utils.audio_utils import load_audio

sys.path.append(os.path.abspath(os.path.join(get_app_root(), 'f5/src')))
sys.path.append(os.path.abspath(os.path.join(get_app_root(), 'f5/src/f5_tts')))
sys.path.append(os.path.abspath(os.path.join(get_app_root(), 'f5/src/f5_tts/model')))

from f5.src.f5_tts.model import DiT, CFM
from f5.src.f5_tts.infer import utils_infer
from f5.src.f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer


class F5Engine(tts_engine):

    def __init__(self):
        super().__init__()
        print("Setting Up F5 Engine")
        self.engine_name = 'F5'
        self.device = cfg.get(cfg.device)
        self.vocoder = None
        self.cfm_model = None
        self.mode = cfg.get(cfg.f5_mode)
        self.model_cfg = None

    def unload_model(self):
        self.model.to('cpu')
        del self.vocoder
        super().basic_unload_model()
        self.vocoder = None
        del self.model_cfg
        self.model_cfg = None
        del self.cfm_model
        self.cfm_model = None

    def switch_models(self):
        self.model.to('cpu')
        del self.model
        self.model = None
        del self.model_cfg
        self.model_cfg = None


    def load_base_model(self):
        self.load_model()

    def load_model(self):
        logging_utils.logger.debug(f"Loading {self.model_path}")

        if self.mode != cfg.get(cfg.f5_mode):
            self.switch_models()

        if self.vocoder is None:
            self.vocoder = utils_infer.load_vocoder(is_local=True, local_path=os.path.abspath(os.path.join(get_app_root(), 'models', 'F5')), device=self.device)

        if self.is_base:
            ckpt_path = str(os.path.abspath(os.path.join(get_app_root(), 'models', 'F5', 'F5TTS_v1_Base', 'model_1250000.safetensors')))
        else:
            ckpt_path = str(os.path.abspath(self.model_path))

        if cfg.get(cfg.f5_mode) == 'tts':
            self.model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            self.model = utils_infer.load_model(DiT, self.model_cfg, ckpt_path, device=self.device)
            self.mode = cfg.get(cfg.f5_mode)
        else:
            self.mode = cfg.get(cfg.f5_mode)
            ode_method = "euler"
            self.model_cfg = OmegaConf.load(str(os.path.abspath(os.path.join(get_app_root(), 'f5', 'src', 'f5_tts', 'configs', 'F5TTS_v1_Base.yaml'))))
            model_cls = get_class(f"f5_tts.model.{self.model_cfg.model.backbone}")
            model_arc = self.model_cfg.model.arch

            dataset_name = self.model_cfg.datasets.name
            tokenizer = self.model_cfg.model.tokenizer

            mel_spec_type = self.model_cfg.model.mel_spec.mel_spec_type
            target_sample_rate = self.model_cfg.model.mel_spec.target_sample_rate
            n_mel_channels = self.model_cfg.model.mel_spec.n_mel_channels
            hop_length = self.model_cfg.model.mel_spec.hop_length
            win_length = self.model_cfg.model.mel_spec.win_length
            n_fft = self.model_cfg.model.mel_spec.n_fft

            vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

            # Model
            self.cfm_model = CFM(
                transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
                mel_spec_kwargs=dict(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
                odeint_kwargs=dict(
                    method=ode_method,
                ),
                vocab_char_map=vocab_char_map,
            ).to(self.device)

            self.model = utils_infer.load_checkpoint(self.cfm_model, ckpt_path, device=self.device, use_ema=True)


    def generate_audio(self, text, transcript=None, voice=None, language='en', output_file=None, streaming=False, start_time=None, end_time=None):

        if self.mode != cfg.get(cfg.f5_mode):
            self.load_model()

        # Load the audio
        audio, sr = librosa.load(
            voice, sr=44100
        )
        silence_duration = 1  # seconds
        silence = np.zeros(int(silence_duration * sr))
        new_audio = np.concatenate([audio, silence])
        sf.write(voice, new_audio, sr)

        if self.mode == 'tts':
            self.inference(text, transcript['transcript'], voice, language, output_file, streaming)
        else:
            self.edit_inference(text, transcript, voice, language, output_file, streaming, start_time=start_time, end_time=end_time)

        rvc_enabled = cfg.get(cfg.rvc_enabled)
        if rvc_enabled and self.rvc_model:
            self.run_rvc(output_file)

        rs_data = load_audio(output_file, 44100)
        sf.write(output_file, rs_data, 44100, subtype='PCM_16')

    @torch.no_grad()
    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False, cross_fade_duration=0.15, nfe_step=32, speed=1, remove_silence=False):
        ref_audio, ref_text = utils_infer.preprocess_ref_audio_text(str(os.path.abspath(os.path.join(get_app_root(), str(voice)))), transcript)

        final_wave, final_sample_rate, combined_spectrogram = utils_infer.infer_process(
            ref_audio,
            ref_text,
            text.strip(),
            self.model,
            self.vocoder,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            speed=speed,
            device=self.device,
        )

        sf.write(output_file, final_wave, final_sample_rate)

        # Remove silence
        if remove_silence:
            utils_infer.remove_silence_for_generated_wav(output_file)
            final_wave, _ = torchaudio.load(output_file)
            final_wave = final_wave.squeeze().cpu().numpy()
            sf.write(output_file, final_wave, final_sample_rate)


    @torch.no_grad()
    def edit_inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False, cross_fade_duration=0.15, nfe_step=32, speed=1, remove_silence=False, start_time=0, end_time=0):
        cfg_strength = 2.0
        sway_sampling_coef = -1.0
        target_rms = 0.1

        tokenizer = self.model_cfg.model.tokenizer
        target_sample_rate = self.model_cfg.model.mel_spec.target_sample_rate
        hop_length = self.model_cfg.model.mel_spec.hop_length

        target_transcript = ""
        target_text = text.strip()

        padding = 0
        previous_word = None

        for word in transcript["words_info"]:
            if word["start"] < start_time:
                target_transcript += word["word"] + (" " if word["word"][-1] != " " else "")
                previous_word = word
            else:
                if previous_word is not None:
                    padding = (previous_word["end"] - word["start"]) / 2
                break
        target_transcript += f"{target_text}"
        for word in transcript["words_info"]:
            if word["end"] > end_time:
                target_transcript += (" " if word["word"][-1] != " " else "") + word["word"]

        logging_utils.logger.debug(f"target_transcript {target_transcript}")

        audio_to_edit = os.path.abspath(voice)

        audio, sr = librosa.load(
            audio_to_edit,
            sr=target_sample_rate
        )

        if audio.ndim == 1:
            audio = torch.from_numpy(audio).unsqueeze(0)
        else:
            audio = torch.from_numpy(audio.T)

        # RMS normalization
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < 1e-6:  # Prevent div by zero
            audio = torch.zeros_like(audio)
        elif rms < target_rms:
            audio = audio * (target_rms / rms)

        # Audio editing logic
        parts_to_edit = [[start_time, end_time]]

        speed_factor = (cfg.get(cfg.f5_speed) / 10.0)

        fix_duration = [math.ceil(end_time - start_time) * speed_factor]

        offset = 0
        audio_ = torch.zeros(1, 0)
        edit_mask = torch.zeros(1, 0, dtype=torch.bool)

        for part in parts_to_edit:
            start, end = part
            part_dur = end - start if fix_duration is None else fix_duration.pop(0)
            part_dur_samples = math.ceil(part_dur * target_sample_rate)
            start_samples = math.ceil(start * target_sample_rate)

            audio_ = torch.cat((
                audio_,
                audio[:, round(offset):round(start_samples)],
                torch.zeros(1, part_dur_samples)
            ), dim=-1)

            edit_mask = torch.cat((
                edit_mask,
                torch.ones(1, round((start_samples - offset) / hop_length), dtype=torch.bool),
                torch.zeros(1, round(part_dur_samples / hop_length), dtype=torch.bool)
            ), dim=-1)

            offset = end * target_sample_rate

        #audio = torch.cat((audio_, audio[:, round(offset):]), dim=-1)
        edit_mask = F.pad(edit_mask, (0, audio.shape[-1] // hop_length - edit_mask.shape[-1] + 1), value=True)

        audio = audio.to(self.device)
        edit_mask = edit_mask.to(self.device)

        # Text
        text_list = [target_transcript]
        if tokenizer == "pinyin":
            final_text_list = convert_char_to_pinyin(text_list)
        else:
            final_text_list = [text_list]
        print(f"text  : {text_list}")
        print(f"pinyin: {final_text_list}")

        # Duration
        ref_audio_len = 0
        duration = audio.shape[-1] // hop_length

        # Inference
        with torch.inference_mode():
            generated, trajectory = self.model.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                edit_mask=edit_mask,
            )

            # Final result
            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            gen_mel_spec = generated.permute(0, 2, 1)
            generated_wave = self.vocoder.decode(gen_mel_spec).cpu()

            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            torchaudio.save(output_file, generated_wave, target_sample_rate)
