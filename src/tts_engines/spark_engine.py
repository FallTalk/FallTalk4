import os
import sys

import numpy as np

from src.enums.engine_type import EngineType
from src.config.config import cfg
from src.tts_engines.tts_engine import tts_engine
from src.utils.filesystem_utils import get_app_code_root, get_app_root

sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'spark')))
sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'spark', 'sparktts')))
sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'spark', 'sparktts', 'utils')))
sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'spark', 'sparktts', 'models')))
sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'spark', 'sparktts', 'modules')))

import re
import torch
from typing import Tuple
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from third_party.spark.sparktts.models.audio_tokenizer import BiCodecTokenizer

class SparkEngine(tts_engine):
    def __init__(self):
        super().__init__()
        self.engin_type = EngineType.SPARK
        self.engine_name = self.engin_type.value
        self.model_type = "pth"
        self.model = None
        self.device = cfg.get(cfg.device)
        self.tokenizer = None
        self.audio_tokenizer = None
        
    def load_model(self):
        if self.is_base:
            self.tokenizer = AutoTokenizer.from_pretrained(str(os.path.abspath(os.path.join(get_app_root(), 'models', 'Spark', '0.5B' "LLM"))))
            self.model = AutoModelForCausalLM.from_pretrained(str(os.path.abspath(os.path.join(get_app_root(), 'models', 'Spark', '0.5B' "LLM"))))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

        self.audio_tokenizer = BiCodecTokenizer(Path(os.path.abspath(os.path.join(get_app_root(), 'models', 'Spark', '0.5B' ))), device=self.device)
        self.audio_tokenizer.model.to(self.device)
        self.model.to(self.device)

    def unload_model(self):
        self.basic_unload_model()
        del self.tokenizer
        self.tokenizer = None
        del self.audio_tokenizer
        self.audio_tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_prompt(self,
            text: str,
            prompt_speech_path: Path,
            prompt_text: str = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        Process input for voice cloning.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.

        ReturI:
            Tuple[str, torch.Tensor]: Input prompt; global tokens
        """
        global_token_ids = None

        # Prepare the input tokens for the model
        if prompt_text is not None:
            global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(
                prompt_speech_path
            )
            global_tokens = "".join(
                [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
            )

            semantic_tokens = "".join(
                [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
            )
            inputs = [
                "<|task_tts|>",
                "<|start_content|>",
                prompt_text + " ",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        else:
            inputs = [
                "<|task_tts|>",
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>"
            ]

        inputs = "".join(inputs)

        return inputs, global_token_ids


    @torch.no_grad()
    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False, speaker=None, start_time=None, end_time=None):
        if not transcript and not voice:
            text = f"{speaker}: " + text if speaker else text

        prompt, global_token_ids = self.process_prompt(
            text, voice, transcript
        )

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        # Generate speech using the model
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=cfg.get(cfg.spark_max_new_tokens),
            do_sample=True,
            top_k=cfg.get(cfg.spark_top_k),
            top_p=cfg.get(cfg.spark_top_p) / 100.0,
            temperature=cfg.get(cfg.spark_temperature) / 100.0,
            eos_token_id=self.tokenizer.eos_token_id,  # Stop token
            pad_token_id=self.tokenizer.pad_token_id  # Use models pad token id
        )

        generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]

        predicts_text = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        # print(f"\nGenerated Text (for parsing):\n{predicts_text}\n") # Debugging

        # Extract semantic token IDs using regex
        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
        if not semantic_matches:
            print("Warning: No semantic tokens found in the generated output.")
            # Handle appropriately - perhaps return silence or raise error
            return np.array([], dtype=np.float32)

        pred_semantic_ids = torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)  # Add batch dim

        # Extract global token IDs using regex (assuming controllable mode also generates these)
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)
        if not global_matches:
            print(
                "Warning: No global tokens found in the generated output (controllable mode). Might use defaults or fail.")
            pred_global_ids = torch.zeros((1, 1), dtype=torch.long)
        else:
            pred_global_ids = torch.tensor([int(token) for token in global_matches]).long().unsqueeze(0)  # Add batch dim

        pred_global_ids = pred_global_ids.unsqueeze(0)  # Shape becomes (1, 1, N_global)

        if global_token_ids is not None:
            wav_np = self.audio_tokenizer.detokenize(
                global_token_ids.to(self.device).squeeze(0),
                pred_semantic_ids.to(self.device)  # Shape (1, N_semantic)
            )
        else:
            wav_np = self.audio_tokenizer.detokenize(
                pred_global_ids.to(self.device).squeeze(0),  # Shape (1, N_global)
                pred_semantic_ids.to(self.device)  # Shape (1, N_semantic)
            )

        return wav_np, 16000
        # sf.write(output_file, wav.cpu().numpy(), 16000)