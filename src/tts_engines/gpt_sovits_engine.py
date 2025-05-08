import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# Create bare minimum mock that will satisfy most basic imports
_fake_gradio = types.ModuleType("gradio")
_fake_gradio.__version__ = "3.50.2"  # Some code checks version

# Add dummy classes that do nothing but prevent AttributeErrors
for name in ['Interface', 'Blocks', 'Slider', 'Textbox', 'Dropdown', 'Audio']:
    setattr(_fake_gradio, name, type(name, (), {'__init__': lambda self, *args, **kwargs: None}))

# Add dummy functions
_fake_gradio.inputs = lambda *args, **kwargs: []
_fake_gradio.outputs = lambda *args, **kwargs: []

sys.modules["gradio"] = _fake_gradio

import numpy as np
import soundfile as sf

from src.enums.engine_type import EngineType
from src.utils import logging_utils
from src.utils.filesystem_utils import get_app_code_root

sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS')))
sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS/tools', 'AP_BWE_main')))
sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS/tools', 'AP_BWE_main', 'models')))
sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS/tools')))
sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS/GPT_SoVITS')))
sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS/GPT_SoVITS/AR')))
sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS/GPT_SoVITS/module')))
sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'f5/src')))
sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'f5/src/f5_tts')))
sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'f5/src/f5_tts/model')))

from src.config.config import cfg
from third_party.GPT_SoVITS.GPT_SoVITS import utils
from src.utils.audio_utils import load_audio
from src.tts_engines.tts_engine import tts_engine
from third_party.GPT_SoVITS.GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

import torch

torch.serialization.add_safe_globals([utils.HParams])


@contextmanager
def patch_gpt_sovits_imports():
    """Temporarily patches the Python imports to ensure GPT_SoVITS utils are found"""
    original_sys_path = sys.path.copy()
    gpt_sovits_root = Path(get_app_code_root()) / "third_party" / "GPT_SoVITS" / "GPT_SoVITS"

    # Add to path and force import
    sys.path.insert(0, str(gpt_sovits_root))
    try:
        # Force import the correct utils module
        import third_party.GPT_SoVITS.GPT_SoVITS.utils as gpt_utils
        sys.modules['utils'] = gpt_utils

        # If the module expects 'utils.HParams' specifically
        if not hasattr(gpt_utils, 'HParams'):
            class HParams:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            gpt_utils.HParams = HParams

        yield
    finally:
        # Restore original state
        sys.path = original_sys_path
        if 'utils' in sys.modules:
            del sys.modules['utils']

class GPT_SoVITS_Engine(tts_engine):

    def __init__(self):
        super().__init__()
        self.engin_type = EngineType.GPT_SOVITS
        self.engine_name = self.engin_type.value
        self.device = cfg.get(cfg.device)
        self.is_half = cfg.get(cfg.low_vram_gpt_sovits)
        self.pipeline: Optional[TTS] = None
        self.config: Optional[TTS_Config] = None
        self.cut_method = {
            "No Slice": "cut0",
            "Slice every 4 sentences": "cut1",
            "Slice per 50 characters": "cut2",
            "Slice by Chinese punct": "cut3",
            "Slice by English punct": "cut4",
            "Slice by every punct": "cut5",
        }

    def unload_model(self):
        if self.pipeline:
            del self.pipeline.sr_model
            del self.pipeline.t2s_model
            del self.pipeline.bert_model
            del self.pipeline.cnhuhbert_model
            del self.pipeline.vocoder
            del self.pipeline.vits_model
            del self.pipeline
            self.pipeline: Optional[TTS] = None
        if self.config:
            del self.config
            self.config: Optional[TTS_Config] = None

    @torch.no_grad()
    def generate_audio(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        self.inference(text=text, voice=voice, transcript=transcript, language=language, output_file=output_file, streaming=streaming)
        if cfg.get(cfg.rvc_enabled) and self.rvc_model:
            self.run_rvc(output_file)

        rs_data = load_audio(output_file, 44100)
        sf.write(output_file, rs_data, 44100, subtype='PCM_16')

    def get_config(self):
        configs: dict = {}

        if self.model_engine_version == '1' or self.model_engine_version == '2':
            if self.is_base:
                configs: dict = {
                    "v2": {
                        "device": self.device,
                        "is_half": self.is_half,
                        "version": "v2",
                        "t2s_weights_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/v2/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
                        "vits_weights_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/v2/s2G2333k.pth"),
                        "cnhuhbert_base_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/chinese-hubert-base"),
                        "bert_base_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/chinese-roberta-wwm-ext-large"),
                        "languages": ["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"]
                    }
                }
            else:
                configs: dict = {
                    "v2": {
                        "device": self.device,
                        "is_half": self.is_half,
                        "version": "v2",
                        "t2s_weights_path": self.get_model(self.engin_type, "ckpt"),
                        "vits_weights_path": os.path.abspath(self.model_path),
                        "cnhuhbert_base_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/chinese-hubert-base"),
                        "bert_base_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/chinese-roberta-wwm-ext-large"),
                        "languages": ["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"]
                    }
                }
        elif self.model_engine_version == '3' or self.model_engine_version == '4':
            if self.is_base:
                configs: dict = {
                    "v4": {
                        "device": self.device,
                        "is_half": self.is_half,
                        "version": "v4",
                        "t2s_weights_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/v3/s1v3.ckpt"),
                        "vits_weights_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/v4/s2Gv4.pth"),
                        "cnhuhbert_base_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/chinese-hubert-base"),
                        "bert_base_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/chinese-roberta-wwm-ext-large"),
                        "languages": ["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"]
                    }
                }
            else:
                configs: dict = {
                    "v4": {
                        "device": self.device,
                        "is_half": self.is_half,
                        "version": "v4",
                        "t2s_weights_path": self.get_model(self.engin_type, "ckpt"),
                        "vits_weights_path": os.path.abspath(self.model_path),
                        "cnhuhbert_base_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/chinese-hubert-base"),
                        "bert_base_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/chinese-roberta-wwm-ext-large"),
                        "languages": ["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"]
                    }
                }

        return configs

    def load_model(self):
        with patch_gpt_sovits_imports():
            self.config = TTS_Config(self.get_config())
            self.pipeline = TTS(self.config)

    @torch.no_grad()
    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        logging_utils.logger.debug("Generating Audio...")

        inputs = {
            "text": text,
            "text_lang": language,
            "ref_audio_path": voice,
            "prompt_text": transcript,
            "prompt_lang": language,
            "top_k": cfg.get(cfg.top_k_gpt_sovits),
            "top_p": (cfg.get(cfg.top_p_gpt_sovits) / 100.0),
            "temperature": (cfg.get(cfg.temperature_gpt_sovits) / 100.0),
            "sample_steps": 32,
            "seed": -1,
            "speed_factor": (cfg.get(cfg.speed_gpt_sovits) / 100.0),
            "text_split_method": self.cut_method.get(cfg.get(cfg.slice_mode), "cut0")
        }

        gen = self.pipeline.run(inputs)
        audio_data = []
        sample_rate = None

        for sr, fragment in gen:
            sample_rate = sr
            audio_data.append(fragment)

        if audio_data:
            combined_audio = np.concatenate(audio_data)
            sf.write(output_file, combined_audio, sample_rate)
            return output_file
        return None
