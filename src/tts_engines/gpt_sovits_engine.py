from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

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

from src.enums.engine_type import EngineType
from src.utils import logging_utils
from src.utils.filesystem_utils import get_app_code_root

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from third_party.GPT_SoVITS.GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

from src.config.config import cfg
from src.tts_engines.tts_engine import tts_engine
import torch

@contextmanager
def patch_gpt_sovits_imports():
    """Full patching solution that handles both HParams and DiT method signatures"""
    original_sys_path = sys.path.copy()
    gpt_sovits_root = Path(get_app_code_root()) / "third_party" / "GPT_SoVITS" / "GPT_SoVITS"
    f5_model_path = gpt_sovits_root / "f5_tts" / "model"

    modules_to_clear = []
    try:

        sys.path.insert(0, os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS')))
        sys.path.insert(0, os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS/GPT_SoVITS')))
        sys.path.insert(0, os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS/GPT_SoVITS/eres2net')))
        sys.path.insert(0, os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS/GPT_SoVITS/AR')))
        sys.path.insert(0, os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS/GPT_SoVITS/module')))
        sys.path.insert(0, os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS/tools', 'AP_BWE_main')))
        sys.path.insert(0, os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS/tools', 'AP_BWE_main', 'models')))
        sys.path.insert(0, os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS/tools')))
        sys.path.insert(0, os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS/tools')))
        sys.path.insert(0, os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS/f5_tts')))
        sys.path.insert(0, os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'GPT_SoVITS/f5_tts/model')))

        # Clean module cache aggressively
        modules_to_clear = [
            'third_party.f5.src.f5_tts.model.backbones.dit',
            'f5_tts.model.backbones.dit',
            'f5_tts.model.backbones',
            'f5_tts.model',
            'utils',
        ]

        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

        # Prioritize our custom implementation's path
        sys.path.insert(0, str(f5_model_path / "backbones"))
        sys.path.insert(0, str(f5_model_path))
        sys.path.insert(0, str(gpt_sovits_root))

        import third_party.GPT_SoVITS.GPT_SoVITS.f5_tts.model as f5_tts_model
        sys.modules['f5_tts.model'] = f5_tts_model

        import third_party.GPT_SoVITS.GPT_SoVITS.f5_tts.model.backbones as backbones
        sys.modules['f5_tts.model.backbones'] = backbones

        import third_party.GPT_SoVITS.GPT_SoVITS.f5_tts.model.backbones.dit as dit_models
        sys.modules['f5_tts.model.backbones.dit'] = dit_models

        # Patch HParams first
        import third_party.GPT_SoVITS.GPT_SoVITS.utils as gpt_utils
        sys.modules['utils'] = gpt_utils

        # Now handle DiT implementation
        from third_party.GPT_SoVITS.GPT_SoVITS.f5_tts.model.backbones.dit import DiT as CorrectDiT

        # Patch into all relevant modules
        import third_party.GPT_SoVITS.GPT_SoVITS.module.models as models
        models.DiT = CorrectDiT

        # Ensure torch's JIT doesn't cache old implementations
        torch._C._jit_clear_class_registry()

        yield
    finally:
        sys.path = original_sys_path
        # Clear module cache again to prevent side effects
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

class GPT_SoVITS_Engine(tts_engine):

    def __init__(self):
        super().__init__()
        self.engin_type = EngineType.GPT_SOVITS
        self.engine_name = self.engin_type.value
        self.device = cfg.get(cfg.device)
        self.is_half = cfg.get(cfg.low_vram_gpt_sovits)
        self.pipeline: Optional['TTS'] = None
        self.config: Optional['TTS_Config'] = None
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
            self.pipeline: Optional['TTS'] = None
        if self.config:
            del self.config
            self.config: Optional['TTS_Config'] = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_audio(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False, speaker=None):
        with patch_gpt_sovits_imports():
            # Get audio data and sample rate from inference
            audio_data, sample_rate = self.inference(text=text, voice=voice, transcript=transcript, language=language, output_file=output_file, streaming=streaming)

            self.process_audio(audio_data, sample_rate, output_file)

    def get_config(self):
        configs: dict = {}
        if not self.is_base and (self.model_engine_version == '1' or self.model_engine_version == '2'):
            configs: dict = {
                "version": "v2",
                "v2": {
                    "device": self.device,
                    "is_half": self.is_half,
                    "version": "v2",
                    "t2s_weights_path": self.get_model(self.engin_type, "ckpt", shared_model_name=self.shared_model_name),
                    "vits_weights_path": os.path.abspath(self.model_path),
                    "cnhuhbert_base_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/chinese-hubert-base"),
                    "bert_base_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/chinese-roberta-wwm-ext-large"),
                    "languages": ["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"]
                }
            }
        elif not self.is_base and self.model_engine_version == 'v2ProPlus':
            configs: dict = {
                "version": "v2ProPlus",
                "custom": {
                    "device": self.device,
                    "is_half": self.is_half,
                    "version": "v2ProPlus",
                    "t2s_weights_path": self.get_model(self.engin_type, "ckpt",  shared_model_name=self.shared_model_name),
                    "vits_weights_path": os.path.abspath(self.model_path),
                    "cnhuhbert_base_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/chinese-hubert-base"),
                    "bert_base_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/chinese-roberta-wwm-ext-large"),
                    "languages": ["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"]
                }
            }
        elif self.is_base:
            configs: dict = {
                "version": "v2ProPlus",
                "custom": {
                    "device": self.device,
                    "is_half": self.is_half,
                    "version": "v2ProPlus",
                    "t2s_weights_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/v3/s1v3.ckpt"),
                    "vits_weights_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/v2Pro/s2Gv2ProPlus.pth"),
                    "cnhuhbert_base_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/chinese-hubert-base"),
                    "bert_base_path": os.path.join(get_app_code_root(), "models/GPT_SoVITS/chinese-roberta-wwm-ext-large"),
                    "languages": ["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"]
                }
            }

        return configs

    def load_model(self):

        with patch_gpt_sovits_imports():
            from third_party.GPT_SoVITS.GPT_SoVITS import utils
            from third_party.GPT_SoVITS.GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
            from third_party.GPT_SoVITS.GPT_SoVITS.eres2net.ERes2NetV2 import ERes2NetV2
            from third_party.GPT_SoVITS.GPT_SoVITS.eres2net import kaldi as Kaldi

            import os, torch
            sv_path = os.path.join(get_app_code_root(), "models/GPT_SoVITS/sv/pretrained_eres2netv2w24s4ep4.ckpt")

            class SV_Overide:
                def __init__(self, device, is_half):
                    pretrained_state = torch.load(sv_path, map_location='cpu', weights_only=False)
                    embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
                    embedding_model.load_state_dict(pretrained_state)
                    embedding_model.eval()
                    self.embedding_model = embedding_model
                    if not is_half:
                        self.embedding_model = self.embedding_model.to(device)
                    else:
                        self.embedding_model = self.embedding_model.half().to(device)
                    self.is_half = is_half

                def compute_embedding3(self, wav):
                    with torch.no_grad():
                        if self.is_half == True: wav = wav.half()
                        feat = torch.stack(
                            [Kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for wav0
                             in wav])
                        sv_emb = self.embedding_model.forward3(feat)
                    return sv_emb

            torch.serialization.add_safe_globals([utils.HParams])

            class TTSOverride(TTS):
                def __init__(self, configs: Union[dict, str, TTS_Config]):
                    with patch_gpt_sovits_imports():
                        super().__init__(configs)

                def init_sv_model(self):
                    with patch_gpt_sovits_imports():
                        if self.sv_model is not None:
                            return
                        self.sv_model = SV_Overide(self.configs.device, self.configs.is_half)

                def init_vocoder(self, version: str):
                    with patch_gpt_sovits_imports():
                        if version == "v4":
                            if self.vocoder is not None and self.vocoder.__class__.__name__ == "Generator":
                                return
                            if self.vocoder is not None:
                                self.vocoder.cpu()
                                del self.vocoder
                                self.empty_cache()
                            from third_party.GPT_SoVITS.GPT_SoVITS.module.models import Generator

                            self.vocoder = Generator(
                                initial_channel=100,
                                resblock="1",
                                resblock_kernel_sizes=[3, 7, 11],
                                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                                upsample_rates=[10, 6, 2, 2, 2],
                                upsample_initial_channel=512,
                                upsample_kernel_sizes=[20, 12, 4, 4, 4],
                                gin_channels=0, is_bias=True
                            )
                            self.vocoder.remove_weight_norm()
                            state_dict_g = torch.load(os.path.join(get_app_code_root(), "models/GPT_SoVITS/v4/vocoder.pth"), map_location="cpu")
                            print("loading vocoder", self.vocoder.load_state_dict(state_dict_g))

                            self.vocoder_configs["sr"] = 48000
                            self.vocoder_configs["T_ref"] = 500
                            self.vocoder_configs["T_chunk"] = 1000
                            self.vocoder_configs["upsample_rate"] = 480
                            self.vocoder_configs["overlapped_len"] = 12

                        self.vocoder = self.vocoder.eval()
                        if self.configs.is_half == True:
                            self.vocoder = self.vocoder.half().to(self.configs.device)
                        else:
                            self.vocoder = self.vocoder.to(self.configs.device)

            self.config = TTS_Config(self.get_config())
            self.pipeline = TTSOverride(self.config)

    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        with patch_gpt_sovits_imports():
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
                # sf.write(output_file, combined_audio, sample_rate)
                return combined_audio, sample_rate
            else:
                return None, None
