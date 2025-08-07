import os
import sys

import torch

from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.tts_engines.tts_engine import tts_engine
from src.utils import logging_utils
from src.utils.filesystem_utils import get_app_code_root, get_app_root

# Add Higgs paths to sys.path
sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'higgs')))

from third_party.higgs.boson_multimodal.data_types import Message, ChatMLSample, AudioContent
from third_party.higgs.boson_multimodal.model.higgs_audio import HiggsAudioModel
from third_party.higgs.boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from third_party.higgs.boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from third_party.higgs.boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from third_party.higgs.boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import StaticCache
from dataclasses import asdict

class HiggsTtsEngine(tts_engine):
    def __init__(self):
        super().__init__()
        self.engin_type = EngineType.HIGGS
        self.engine_name = self.engin_type.value
        self.model_type = "safetensors"
        self.model = None
        self.device = cfg.get(cfg.device)
        self.tokenizer = None
        self.audio_tokenizer = None
        self.kv_caches = None
        self.collator = None
        self.use_static_kv_cache = True
        self.max_new_tokens = 2048
        self.kv_cache_lengths = [1024, 4096, 8192]

    def load_model(self):
        logging_utils.logger.info(f"Loading Higgs TTS model from {self.model_path}")

        # Determine model paths
        if self.is_base:
            # Use base model paths
            tokenizer_path = os.path.abspath(os.path.join(get_app_root(), 'models', 'Higgs', 'v2', 'tokenizer'))
            final_path = os.path.abspath(os.path.join(get_app_root(), 'models', 'Higgs', 'v2'))
            config_path = os.path.abspath(os.path.join(get_app_root(), 'models', 'Higgs', 'v2'))
        else:
            # Use character-specific model path
            tokenizer_path = os.path.join(self.model_path)
            final_path = os.path.join(self.model_path)
            config_path = os.path.join(self.model_path)

        # Check if models exist locally, otherwise fall back to HF
        logging_utils.logger.info(f"Loading Higgs audio tokenizer from local path: {tokenizer_path}")
        self.audio_tokenizer = load_higgs_audio_tokenizer(tokenizer_path, device=self.device)

        # Load the model
        logging_utils.logger.info(f"Loading Higgs model from local path: {final_path}")
        self.model = HiggsAudioModel.from_pretrained(
            final_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )

        self.model.eval()

        logging_utils.logger.info(f"Loading Higgs tokenizer and config from local path: {config_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(config_path)
        self.config = AutoConfig.from_pretrained(config_path)

        # Initialize collator
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=None,
            audio_in_token_id=self.config.audio_in_token_idx,
            audio_out_token_id=self.config.audio_out_token_idx,
            audio_stream_bos_id=self.config.audio_stream_bos_id,
            audio_stream_eos_id=self.config.audio_stream_eos_id,
            encode_whisper_embed=self.config.encode_whisper_embed,
            pad_token_id=self.config.pad_token_id,
            return_audio_in_tokens=self.config.encode_audio_in_tokens,
            use_delay_pattern=self.config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=self.config.audio_num_codebooks,
        )


    def unload_model(self):
        self.basic_unload_model()
        del self.tokenizer
        self.tokenizer = None
        del self.audio_tokenizer
        self.audio_tokenizer = None
        del self.collator
        self.collator = None

        if self.kv_caches:
            del self.kv_caches
            self.kv_caches = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_generation_context(self, text, voice, transcript=None):
        """Prepare the context for generation with reference audio."""
        messages = []
        audio_ids = []

        # Create system message
        system_message = Message(
            role="system",
            content="Generate audio following instruction."
        )
        messages.append(system_message)

        # Process reference audio if provided
        if voice is not None and os.path.exists(voice):
            # Encode the reference audio
            audio_tokens = self.audio_tokenizer.encode(voice)
            audio_ids.append(audio_tokens)

            # Add reference audio as a message pair
            if transcript:
                messages.append(
                    Message(
                        role="user",
                        content=transcript
                    )
                )
                messages.append(
                    Message(
                        role="assistant",
                        content=AudioContent(
                            audio_url=voice
                        )
                    )
                )

        return messages, audio_ids

    @torch.inference_mode()
    def generate_audio(self, text, transcript=None, voice=None, language='en', output_file=None, streaming=False, speaker=None):
        """Generate audio using Higgs TTS model."""
        try:
            # Prepare generation context
            messages, audio_ids = self.prepare_generation_context(text, voice, transcript)

            # Add the text to generate as a user message
            messages.append(
                Message(
                    role="user",
                    content=text
                )
            )

            # Prepare the input for the model
            chatml_sample = ChatMLSample(messages=messages)
            input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample, self.tokenizer)
            postfix = self.tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False
            )
            input_tokens.extend(postfix)

            # Prepare the sample for the model
            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=torch.concat([ele.cpu() for ele in audio_ids], dim=1)
                if audio_ids
                else None,
                audio_ids_start=torch.cumsum(
                    torch.tensor([0] + [ele.shape[1] for ele in audio_ids], dtype=torch.long), dim=0
                )
                if audio_ids
                else None,
                audio_waveforms_concat=None,
                audio_waveforms_start=None,
                audio_sample_rate=None,
                audio_speaker_indices=None,
            )

            # Collate the sample
            batch_data = self.collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self.device)

            # Generate audio
            outputs = self.model.generate(
                **batch,
                max_new_tokens=self.max_new_tokens,
                use_cache=False,
                do_sample=True,
                temperature=float(cfg.get(cfg.higgs_temperature) / 100.0),
                top_k=cfg.get(cfg.higgs_top_k),
                top_p=(cfg.get(cfg.higgs_top_p) / 100.0),
                ras_win_len=7,
                ras_win_max_num_repeat=2,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self.tokenizer,
                seed=cfg.get(cfg.seed),
            )

            # Process the output
            audio_out_ids_l = []
            for ele in outputs[1]:
                audio_out_ids = ele
                if self.config.use_delay_pattern:
                    audio_out_ids = revert_delay_pattern(audio_out_ids)
                audio_out_ids_l.append(audio_out_ids.clip(0, self.audio_tokenizer.codebook_size - 1)[:, 1:-1])

            audio_out_ids = torch.concat(audio_out_ids_l, dim=1)

            # Fix MPS compatibility: detach and move to CPU before decoding
            if audio_out_ids.device.type == "mps":
                audio_out_ids_cpu = audio_out_ids.detach().cpu()
            else:
                audio_out_ids_cpu = audio_out_ids

            # Decode the audio
            audio_data = self.audio_tokenizer.decode(audio_out_ids_cpu.unsqueeze(0))[0, 0]
            sample_rate = 24000

            # Process the audio (apply effects, save to file)
            self.process_audio(audio_data, sample_rate, output_file)

            return audio_data, sample_rate

        except Exception as e:
            logging_utils.logger.exception(f"Error generating audio with Higgs TTS: {e}")
            raise
