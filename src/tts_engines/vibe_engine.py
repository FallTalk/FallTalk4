import os
import re
import time
from typing import Optional, Tuple, List

import torch
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.tts_engines.tts_engine import tts_engine
from src.utils import torch_utils
from src.utils.filesystem_utils import get_app_root

class VibeEngine(tts_engine):

    def __init__(self):
        super().__init__()
        self.engine_type = EngineType.VIBE
        self.engine_name = self.engine_type.value
        print(f"Setting Up {self.engine_name} Engine")
        self.device = cfg.get(cfg.device)
        self.processor: Optional[VibeVoiceProcessor] = None


    def load_model(self):
        print("Loading Vibe Model")

        if self.is_base:
            base_path = str(os.path.abspath(os.path.join(get_app_root(), 'models', 'Vibe', str(cfg.get(cfg.vibe_mode)))))
            self.processor = VibeVoiceProcessor.from_pretrained(base_path)
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                base_path,
                torch_dtype=torch_utils.get_compute_dtype(),
                device_map=self.device,
                attn_implementation="flash_attention_2"
            )
        else:
            self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=torch_utils.get_compute_dtype(),
                device_map=self.device,
                attn_implementation="flash_attention_2"
            )

        self.model.eval()

        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )


    def unload_model(self):
        self.basic_unload_model()
        del self.processor
        self.processor = None

    def parse_txt_script(self, txt_content: str) -> Tuple[List[str], List[str]]:
        """
        Parse txt script content and extract speakers and their text
        Fixed pattern: Speaker 1, Speaker 2, Speaker 3, Speaker 4
        Returns: (scripts, speaker_numbers)
        """
        lines = txt_content.strip().split('\n')
        scripts = []
        speaker_numbers = []

        # Pattern to match "Speaker X:" format where X is a number
        speaker_pattern = r'^Speaker\s+(\d+):\s*(.*)$'

        current_speaker = None
        current_text = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = re.match(speaker_pattern, line, re.IGNORECASE)
            if match:
                # If we have accumulated text from previous speaker, save it
                if current_speaker and current_text:
                    scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                    speaker_numbers.append(current_speaker)

                # Start new speaker
                current_speaker = match.group(1).strip()
                current_text = match.group(2).strip()
            else:
                # Continue text for current speaker
                if current_text:
                    current_text += " " + line
                else:
                    current_text = line

        # Don't forget the last speaker
        if current_speaker and current_text:
            scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
            speaker_numbers.append(current_speaker)

        return scripts, speaker_numbers


    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False, speaker=None, start_time=None, end_time=None):
        text = f"Speaker 0: {text}"

        scripts, speaker_numbers = self.parse_txt_script(text)

        inputs = self.processor(
            text=[text],  # Wrap in list for batch processing
            voice_samples=[voice],  # Wrap in list for batch processing
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        self.model.set_ddpm_inference_steps(num_steps=cfg.get(cfg.vibe_inference_steps))

        start_time = time.time()

        cfg_scale = cfg.get(cfg.vibe_cfg_scale) / 10
        do_sample = cfg.get(cfg.vibe_dosmaple)
        temperature = cfg.get(cfg.vibe_temperature) / 100
        top_p = cfg.get(cfg.vibe_top_p) / 100
        # Generate audio
        with torch.no_grad():
            if not do_sample:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    do_sample=False
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p
                )

        generation_time = time.time() - start_time
        print(f"Generation time: {generation_time:.2f} seconds")

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            # Assuming 24kHz sample rate (common for speech synthesis)
            sample_rate = 24000
            audio_samples = outputs.speech_outputs[0].shape[-1] if len(outputs.speech_outputs[0].shape) > 0 else len(
                outputs.speech_outputs[0])
            audio_duration = audio_samples / sample_rate
            rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')

            print(f"Generated audio duration: {audio_duration:.2f} seconds")
            print(f"RTF (Real Time Factor): {rtf:.2f}x")
        else:
            print("No audio output generated")

        # Calculate token metrics
        input_tokens = inputs['input_ids'].shape[1]  # Number of input tokens
        output_tokens = outputs.sequences.shape[1]  # Total tokens (input + generated)
        generated_tokens = output_tokens - input_tokens

        print(f"Prefilling tokens: {input_tokens}")
        print(f"Generated tokens: {generated_tokens}")
        print(f"Total tokens: {output_tokens}")

        return outputs.speech_outputs[0].float().cpu().numpy().flatten(), 24000