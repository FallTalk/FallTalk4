import os
import sys
import time

import numpy as np
import torch
import torchaudio

from src.enums.engine_type import EngineType
from src.config.config import cfg
from src.tts_engines.tts_engine import tts_engine
from src.utils.filesystem_utils import get_app_code_root, get_app_root
from src.utils.logging_utils import logger

sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'fish')))
sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'fish', 'fish_speech')))
sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'fish', 'fish_speech', 'models')))

from third_party.fish.fish_speech.models.text2semantic.inference import init_model as fish_load, generate_long
from third_party.fish.fish_speech.models.dac.inference import load_model as load_dac

class FishSpeechEngine(tts_engine):

    def __init__(self):
        super().__init__()
        print("Setting Up FishSpeech Engine")
        self.engin_type = EngineType.FISH_SPEECH
        self.engine_name = self.engin_type.value
        self.device = cfg.get(cfg.device)
        self.fish = None
        self.decode_one_token = None
        self.dac_model = None


    def load_model(self):
        logger.info("Loading model ...")
        t0 = time.time()

        checkpoint_path = str(os.path.abspath(os.path.join(get_app_root(), 'models', 'Fish', "s1-mini")))
        self.dac_model = load_dac("modded_dac_vq", os.path.join(checkpoint_path, "codec.pth"))

        if self.is_base:
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

    def decode_vq_tokens(self, codes):
        feature_lengths = torch.tensor(
            [codes.shape[1]], device=self.dac_model.device
        )
        return self.dac_model.decode(
            indices=codes[None],
            feature_lengths=feature_lengths,
        )[0].squeeze().float().cpu().numpy()

    @torch.no_grad()
    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        print("Loading Fish Model")

        prompt_audio = None

        if voice:
            audio, sr = torchaudio.load(str(voice))
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
            audio = torchaudio.functional.resample(audio, sr, self.dac_model.sample_rate)

            audios = audio[None].to(self.device)
            logger.info(f"Loaded audio with {audios.shape[2] / self.dac_model.sample_rate:.2f} seconds")

            # VQ Encoder
            audio_lengths = torch.tensor([audios.shape[2]], device=self.device, dtype=torch.long)
            indices, indices_lens = self.dac_model.encode(audios, audio_lengths)
            if indices.ndim == 3:
                indices = indices[0]
            prompt_audio = indices.cpu()

        generator = generate_long(
            model=self.model,
            device=self.device,
            decode_one_token=self.decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=cfg.get(cfg.fish_max_length),
            top_p=(cfg.get(cfg.fish_top_p) / 100.0),
            repetition_penalty=float(cfg.get(cfg.fish_repetition) / 10.0),
            temperature=float(cfg.get(cfg.fish_temperature) / 100.0),
            compile=cfg.get(cfg.fish_use_torch_compile),
            iterative_prompt=cfg.get(cfg.fish_iterative_prompt),
            chunk_length=150,
            prompt_text=transcript,
            prompt_tokens=prompt_audio
        )

        all_audio_chunks = []

        for response in generator:
            if response.action == "sample":
                logger.info(f"Sampled text: {response.text}")
                # Decode VQ codes immediately
                decoded_audio = self.decode_vq_tokens(response.codes)
                all_audio_chunks.append(decoded_audio)

            elif response.action == "next":
                # Optionally log or track segmenting
                logger.debug("Moving to next segment.")
            else:
                logger.error(f"Error: {response}")

        if not all_audio_chunks:
            logger.warning("No audio was generated; nothing to save.")
            return None, None
        else:
            full_audio = np.concatenate(all_audio_chunks, axis=-1)  # 1D mono audio
            return full_audio, self.dac_model.sample_rate
