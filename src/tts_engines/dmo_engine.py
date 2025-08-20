import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.tts_engines.tts_engine import tts_engine
from src.utils import logging_utils
from src.utils.filesystem_utils import get_app_root, get_app_code_root


@contextmanager
def patch_gpt_sovits_imports():
    """Full patching solution that handles both HParams and DiT method signatures"""
    original_sys_path = sys.path.copy()
    gpt_sovits_root = Path(get_app_code_root()) / "third_party" / "DMOSpeech2" / "src"
    f5_model_path = gpt_sovits_root / "f5_tts" / "model"

    modules_to_clear = []
    try:

        sys.path.insert(0, os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'DMOSpeech2')))
        sys.path.insert(0, os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'DMOSpeech2/src')))
        sys.path.insert(0, os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'DMOSpeech2/src/f5_tts')))
        sys.path.insert(0, os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'DMOSpeech2/src/f5_tts/model')))

        # Clean module cache aggressively
        modules_to_clear = [
            'third_party.f5.src.f5_tts.model.backbones.dit',
            'f5_tts.model.backbones.dit',
            'f5_tts.model.backbones',
            'f5_tts.model',
        ]

        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

        # Prioritize our custom implementation's path
        sys.path.insert(0, str(f5_model_path / "backbones"))
        sys.path.insert(0, str(f5_model_path))
        sys.path.insert(0, str(gpt_sovits_root))

        import third_party.DMOSpeech2.src.f5_tts.model as f5_tts_model
        sys.modules['f5_tts.model'] = f5_tts_model

        import third_party.DMOSpeech2.src.f5_tts.model.backbones as backbones
        sys.modules['f5_tts.model.backbones'] = backbones

        import third_party.DMOSpeech2.src.f5_tts.model.backbones.dit as dit_models
        sys.modules['f5_tts.model.backbones.dit'] = dit_models

        # Now handle DiT implementation
        from third_party.DMOSpeech2.src.f5_tts.model.backbones.dit import DiT as CorrectDiT

        # Ensure torch's JIT doesn't cache old implementations
        torch._C._jit_clear_class_registry()

        yield
    finally:
        sys.path = original_sys_path
        # Clear module cache again to prevent side effects
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]


class DMSpeech2Engine(tts_engine):

    def __init__(self):
        super().__init__()
        self.engine_type = EngineType.DMOSPEECH2
        self.engine_name = self.engine_type.value
        self.device = cfg.get(cfg.device)
        self.model_type = "pt"

    def clean(self):
        self.unload_model()

    def unload_model(self):
        super().basic_unload_model()


    def load_model(self):
        with patch_gpt_sovits_imports():
            from third_party.DMOSpeech2.src.infer import DMOInference

            logging_utils.logger.debug(f"Loading {self.model_path}")
            duration_checkpoint = os.path.abspath(os.path.join(get_app_root(), 'models', 'DMOSpeech2', 'v2', "model_1500.pt"))

            if self.is_base:
                student_checkpoint = os.path.abspath(os.path.join(get_app_root(), 'models', 'DMOSpeech2', 'v2', "model_85000.pt"))
            else:
                student_checkpoint = self.model_path

            self.model = DMOInference(
                student_checkpoint_path=student_checkpoint,
                duration_predictor_path=duration_checkpoint,
                device=self.device,
                model_type="F5TTS_Base"
            )


    @torch.no_grad()
    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False, speaker=None, start_time=None, end_time=None):
        return self.model.generate(
            gen_text=text,
            audio_path=voice,
            prompt_text=transcript,
            teacher_steps=cfg.get(cfg.dmo_speech2_teacher_steps),
            teacher_stopping_time=cfg.get(cfg.dmo_speech2_teacher_stopping_time) / 100.0,
            student_start_step=cfg.get(cfg.dmo_speech2_student_start_step),
            temperature=cfg.get(cfg.dmo_speech2_temperature) / 100.0,
            verbose=True
        ), 24000
