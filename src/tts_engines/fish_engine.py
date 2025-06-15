import time

import numpy as np
import torchaudio
import librosa

from enums.engine_type import EngineType
from src.config.config import cfg

import sys
import os
import torch
from src.tts_engines.tts_engine import tts_engine
from src.utils.filesystem_utils import get_app_code_root, get_app_root
from src.utils.logging_utils import logger
from src.utils.audio_utils import load_audio

sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'fish/fish_speech')))

from third_party.fish.fish_speech.models.text2semantic.inference import init_model as fish_load, generate_long

class FishSpeechEngine(tts_engine):

    def __init__(self):
        super().__init__()
        print("Setting Up FishSpeech Engine")
        self.engin_type = EngineType.FISH_SPEECH
        self.engine_name = self.engin_type.value
        self.device = cfg.get(cfg.device)
        self.fish = None
        self.decode_one_token = None


    def load_model(self):
        logger.info("Loading model ...")
        t0 = time.time()

        if(self.is_base):
            checkpoint_path = str(os.path.abspath(os.path.join(get_app_root(), 'models', 'FishSpeech', "1.5")))
            self.model, self.decode_one_token = fish_load(
                checkpoint_path, self.device, torch.bfloat16, compile=cfg.get(cfg.fish_use_torch_compile)
            )
        else:
            self.model, self.decode_one_token = fish_load(
                self.model_path, self.device, torch.bfloat16, compile=cfg.get(cfg.fish_use_torch_compile)
            )
        with torch.device(self.device):
            self.model.setup_caches(
                max_batch_size=1,
                max_seq_len=self.model.config.max_seq_len,
                dtype=next(self.model.parameters()).dtype,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

    def generate_audio(self, text, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        audio_data, sample_rate = self.inference(text, transcript, voice, language, output_file, streaming)
        self.process_audio(audio_data, sample_rate, output_file)

    def unload_model(self):
        self.basic_unload_model()
        del self.decode_one_token
        self.decode_one_token = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        print("Loading Fish Model")

        generator = generate_long(
            model=self.model,
            device=self.device,
            decode_one_token=self.decode_one_token,
            text=text,
            num_samples=2,
            max_new_tokens=cfg.get(cfg.fish_max_length),
            top_p=(cfg.get(cfg.fish_top_p) / 100.0),
            repetition_penalty=float(cfg.get(cfg.fish_repetition) / 10.0),
            temperature=float(cfg.get(cfg.fish_temperature) / 100.0),
            compile=cfg.get(cfg.fish_use_torch_compile),
            iterative_prompt=cfg.get(cfg.fish_iterative_prompt),
            chunk_length=150,
            prompt_text=transcript,
        )

        idx = 0
        codes = []
        all_audio_chunks = []

        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)
                logger.info(f"Sampled text: {response.text}")
            elif response.action == "next":
                if codes:
                    all_audio_chunks.append(torch.cat(codes, dim=1).cpu().numpy())
                codes = []
                idx += 1
            else:
                logger.error(f"Error: {response}")

        if not all_audio_chunks:
            logger.warning("No audio was generated; nothing to save.")
        else:
            full_audio = np.concatenateconcatenate(all_audio_chunks, axis=-1)
            return full_audio, self.vqgan_model.spec_transform.sample_rate
            # sf.write(output_file, full_audio, self.vqgan_model.spec_transform.sample_rate)
            # total_duration = full_audio.shape[-1] / self.vqgan_model.spec_transform.sample_rate
            # logger.info(f"Saved combined audio to {output_file}; total duration {total_duration:.2f}s")
