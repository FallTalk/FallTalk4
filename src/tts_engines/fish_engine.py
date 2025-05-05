import time

import numpy as np
import torchaudio

from src.config.config import cfg

import sys
import os
import torch
from src.tts_engines.tts_engine import tts_engine
from src.utils.filesystem_utils import get_app_code_root, get_app_root
from src.utils.logging_utils import logger
from src.utils.audio_utils import load_audio

sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'fish-speech/fish_speech')))

from third_party.fish.fish_speech.models.vqgan import inference as vqgan_inference
from third_party.fish.fish_speech.models.text2semantic.inference import load_model, generate_long
import soundfile as sf

class FishSpeechEngine(tts_engine):

    def __init__(self):
        super().__init__()
        print("Setting Up FishSpeech Engine")
        self.engine_name = 'FishSpeech'
        self.device = cfg.get(cfg.device)
        self.fish = None
        self.vqgan_model = None
        self.decode_one_token = None


    def load_model(self):
        vqgan_path = str(os.path.abspath(os.path.join(get_app_root(), 'models', 'FishSpeech', "1.5" 'firefly-gan-vq-fsq-8x1024-21hz-generator.pth')))
        self.vqgan_model = vqgan_inference.load_model("firefly_gan_vq", vqgan_path, self.device)

        logger.info("Loading model ...")
        t0 = time.time()

        if(self.is_base):
            checkpoint_path = str(os.path.abspath(os.path.join(get_app_root(), 'models', 'FishSpeech', "1.5")))
            self.model, self.decode_one_token = load_model(
                checkpoint_path, self.device, torch.bfloat16, compile=cfg.get(cfg.fish_use_torch_compile)
            )
        else:
            self.model, self.decode_one_token = load_model(
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
        self.inference(text, transcript, voice, language, output_file, streaming)
        rvc_enabled = cfg.get(cfg.rvc_enabled)
        if rvc_enabled and self.rvc_model:
            self.run_rvc(output_file)

        rs_data = load_audio(output_file, 44100)
        sf.write(output_file, rs_data, 44100, subtype='PCM_16')

    def unload_model(self):
        self.basic_unload_model()
        del self.vqgan_model
        del self.decode_one_token
        self.vqgan_model = None
        self.decode_one_token = None

    def numpy_gen(self, input_path):
        logger.info(f"Processing in-place reconstruction of {input_path}")
        # Load audio
        audio, sr = torchaudio.load(str(input_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(
            audio, sr, self.vqgan_model.spec_transform.sample_rate
        )
        audios = audio[None].to(self.device)
        logger.info(
            f"Loaded audio with {audios.shape[2] / self.vqgan_model.spec_transform.sample_rate:.2f} seconds"
        )
        # VQ Encoder
        audio_lengths = torch.tensor([audios.shape[2]], device=self.device, dtype=torch.long)
        indices = self.vqgan_model.encode(audios, audio_lengths)[0][0]

        logger.info(f"Generated indices of shape {indices.shape}")
        # Save indices
        return indices.cpu().numpy()

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
            prompt_tokens=self.numpy_gen(voice),
        )

        idx = 0
        codes = []
        all_audio_chunks = []

        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)
                logger.info(f"Sampled text: {response.text}")
            elif response.action == "next":
                if not codes:
                    logger.info("No codes to process; skipping.")
                else:
                    all_codes = torch.cat(codes, dim=1).to(self.device)

                    feature_lengths = torch.tensor([all_codes.shape[1]], device=self.device)
                    fake_audios, _ = self.vqgan_model.decode(
                        indices=all_codes[None],  # add batch dim
                        feature_lengths=feature_lengths[None]
                    )

                    fake_audio_np = fake_audios[0, 0].float().cpu().numpy()
                    all_audio_chunks.append(fake_audio_np)

                    audio_time = fake_audio_np.shape[-1] / self.vqgan_model.spec_transform.sample_rate
                    logger.info(
                        f"[{idx}] Chunk shape={fake_audio_np.shape}, duration={audio_time:.2f}s"
                    )

                # reset for next segment
                codes = []
                idx += 1
            else:
                logger.error(f"Unknown action: {response}")

        if not all_audio_chunks:
            logger.warning("No audio was generated; nothing to save.")
        else:
            full_audio = np.concatenate(all_audio_chunks, axis=-1)
            sf.write(output_file, full_audio, self.vqgan_model.spec_transform.sample_rate)
            total_duration = full_audio.shape[-1] / self.vqgan_model.spec_transform.sample_rate
            logger.info(f"Saved combined audio to {output_file}; total duration {total_duration:.2f}s")

