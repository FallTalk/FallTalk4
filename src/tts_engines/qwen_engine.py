import os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.tts_engines.tts_engine import tts_engine
from src.utils.audio_utils import load_audio
from src.utils.filesystem_utils import get_app_root
from src.utils import torch_utils

class QwenEngine(tts_engine):

    def __init__(self):
        super().__init__()
        print("Setting Up Qwen3 Engine")
        self.engine_type = EngineType.QWEN3_TTS
        self.engine_name = self.engine_type.value
        self.device = cfg.get(cfg.device)
        self.model = None

    def setup(self, selected_model, rvc=False, base_model=False, model_version=None, is_shared=False, shared_model_name=None, characters=None):
        # Qwen handles model loading slightly differently because it's mostly a base model engine
        if base_model and selected_model is None:
            selected_model = cfg.get(cfg.qwen_model_version)
        
        super().setup(selected_model, rvc, base_model, model_version, is_shared, shared_model_name, characters)

    def load_model(self):
        selected_version = cfg.get(cfg.qwen_model_version)
        model_id = f"Qwen3TTS/Qwen3-TTS-12Hz-{selected_version}"
        
        if self.is_base:
            base_path = os.path.join(get_app_root(), 'models', 'Qwen3TTS', f'Qwen3-TTS-12Hz-{selected_version}')
            if os.path.exists(base_path):
                model_id = str(os.path.abspath(base_path))
        else:
             if self.model_path and os.path.exists(self.model_path):
                 model_id = str(os.path.abspath(self.model_path))

        dtype = torch_utils.get_compute_dtype()
        # Ensure it's bfloat16 or float16 as per typical LLM usage if supported
        if dtype == torch.float32 and self.device != 'cpu':
             dtype = torch.float16 # fallback for GPUs not supporting bf16
        
        # If we are using flash_attention_2, we MUST have a floating point dtype (bf16 or fp16)
        # torch_utils.get_compute_dtype() usually returns one of these if CUDA is available.
        # However, to be extra safe and avoid the warning:
        if dtype == torch.float32 and not self.device == 'cpu':
            # This should have been handled above, but let's be explicit
            dtype = torch.float16

        print(f"Loading Qwen Model: {model_id} with dtype: {dtype}")
        
        # Decide on attention implementation
        attn_impl = "sdpa"
        if torch_utils.supports_bf16():
            attn_impl = "flash_attention_2"
            # Flash Attention 2 requires bf16 or fp16. 
            # If for some reason dtype is still float32 (e.g. CPU or forced), 
            # we should fallback to sdpa to avoid the warning/error.
            if dtype == torch.float32:
                attn_impl = "sdpa"

        self.model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=self.device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )

    def unload_model(self):
        self.basic_unload_model()
        del self.model
        self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False, speaker=None, start_time=None, end_time=None):
        target_lang = cfg.get(cfg.qwen_language)
        selected_version = cfg.get(cfg.qwen_model_version)

        # For Qwen, voice is ref_audio and transcript is ref_text for cloning
        ref_audio = voice
        ref_text = transcript
        
        with torch.inference_mode():
            if "Base" in selected_version:
                # Voice cloning mode
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=target_lang,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
            else:
                # Custom voice mode
                instruct = cfg.get(cfg.qwen_instruct)
                wavs, sr = self.model.generate_custom_voice(
                    text=text,
                    language=target_lang,
                    speaker=speaker if speaker else "Vivian",
                    instruct=instruct if instruct else None,
                )
        
        # wavs[0] because generation methods return a list of wavs
        return wavs[0], sr
