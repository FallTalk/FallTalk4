import glob
import os

import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from falltalk.config import cfg
from tts_engines.tts_engine import tts_engine

from falltalk import falltalkutils

try:
    import deepspeed
    deepspeed_available = True
except ImportError:
    deepspeed_available = False
    pass


class XTTS_Engine(tts_engine):
    def __init__(self):
        super().__init__()
        self.engine_name = 'XTTSv2'

    def clean(self):
        super().clean()
        self.unload_model()

    def unload_model(self):
        super().basic_unload_model()

    @torch.no_grad()
    def generate_audio(self, text=None, voice=None, language=None, output_file=None, streaming=False):
        self.inference(text=text, voice=voice, language=language, output_file=output_file, streaming=streaming)
        if cfg.get(cfg.rvc_enabled) and self.rvc_model:
            self.run_rvc(output_file)

    @torch.no_grad()
    def inference(self, text=None, voice=None, language=None, output_file=None, streaming=False):
        falltalkutils.logger.debug("Generating Audio...")
        if cfg.low_vram and self.device == "cpu":  # If necessary, move the model out of System Ram to VRAM
            self.handle_lowvram_change()

        if os.path.isdir(voice):
            wavs_files = glob.glob(os.path.join(voice, "*.wav"))
        else:
            wavs_files = [voice]

        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=wavs_files,
            gpt_cond_len=self.model.config.gpt_cond_len,
            max_ref_length=self.model.config.max_ref_len,
            sound_norm_refs=self.model.config.sound_norm_refs,
        )
        # Common arguments for both functions
        common_args = {
            "text": text,
            "language": language,
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding,
            "temperature": float(cfg.get(cfg.model_temperature) / 100.0),
            "length_penalty": float(self.model.config.length_penalty),
            "repetition_penalty": float(cfg.get(cfg.model_repetition)),
            "top_k": int(self.model.config.top_k),
            "top_p": float(self.model.config.top_p),
            "speed": float(cfg.get(cfg.speed) / 100.0),
            "enable_text_splitting": True
        }

        # Determine the correct inference function and add streaming specific argument if needed
        inference_func = self.model.inference_stream if streaming else self.model.inference
        if streaming:
            common_args["stream_chunk_size"] = 20

        falltalkutils.logger.debug("Starting...")
        # Call the appropriate function
        output = inference_func(**common_args)
        falltalkutils.logger.debug(f'Done')
        # Convert the NumPy array to a PyTorch tensor
        wav_tensor = torch.tensor(output["wav"])
        # Add a batch dimension
        wav_tensor = wav_tensor.unsqueeze(0)

        torchaudio.save(output_file, wav_tensor, 24000)

        if cfg.get(cfg.low_vram) and self.device == "cuda":
            self.handle_lowvram_change()

    def load_base_model(self):
        self.model_path = os.path.join("models", "XTTSv2", "model.pth")
        self.load_model()

    def load_model(self):
        falltalkutils.logger.debug(f"Loading {self.model_path}")
        config = XttsConfig()
        config_path = os.path.join("models", "XTTSv2", "config.json")
        vocab_path_dir = os.path.join("models", "XTTSv2", "vocab.json")
        speaker_file_path = os.path.join("models", "XTTSv2", "speakers_xtts.json")

        checkpoint_path = os.path.abspath(self.model_path)
        config.load_json(config_path)

        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config,
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path_dir,
            speaker_file_path=speaker_file_path,
            use_deepspeed=cfg.get(cfg.deepspeed_enabled) if deepspeed_available else False,
        )
        self.model.to(self.device)
