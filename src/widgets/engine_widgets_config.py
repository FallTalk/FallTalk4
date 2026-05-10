from __future__ import annotations

from typing import Dict, Type

from src.enums.engine_type import EngineType
from src.help.chatterbox_help import ChatterboxHelp
from src.help.csm_help import CSMHelp
from src.help.dia_help import DIAHelp
from src.help.dmo_speech2_help import DMSpeech2Help
from src.help.f5_help import F5Help
from src.help.fish_help import FishHelp
from src.help.gpt_sovits_help import GPTSoVITSHelp
from src.help.higgs_help import HiggsHelp
from src.help.llasa_help import LlasaHelp
from src.help.moss_tts_help import MossTTSHelp
from src.help.orpheus_help import OrpheusHelp
from src.help.omnivoice_help import OmniVoiceHelp
from src.help.qwen_help import QwenHelp
from src.help.spark_help import SparkHelp
from src.help.styletts2_help import StyleTTS2Help
from src.help.vibe_help import VibeHelp
from src.settings.chatterbox_settings import ChatterboxSettings
from src.settings.csm_settings import CSMSettings
from src.settings.dia_settings import DIASettings
from src.settings.dmo_speech2_settings import DMSpeech2Settings
from src.settings.f5_settings import F5Settings
from src.settings.fish_speech_settings import FishSpeechSettings
from src.settings.gpt_sovits_settings import GPTSoVITSSettings
from src.settings.higgs_settings import HiggsSettings
from src.settings.llasa_settings import LlasaSettings
from src.settings.moss_tts_settings import MossTTSSettings
from src.settings.omnivoice_settings import OmniVoiceSettings
from src.settings.orpheus_settings import OrpheusSettings
from src.settings.qwen_settings import QwenSettings
from src.settings.spark_settings import SparkSettings
from src.settings.styletts2_settings import StyleTTS2Settings
from src.settings.vibe_settings import VibeSettings

# Mapping of engine types to their settings widgets
SETTINGS_WIDGETS: Dict[EngineType, Type] = {
    EngineType.FISH_SPEECH: FishSpeechSettings,
    EngineType.GPT_SOVITS: GPTSoVITSSettings,
    EngineType.LLASA: LlasaSettings,
    EngineType.HIGGS: HiggsSettings,
    EngineType.SPARK: SparkSettings,
    EngineType.ORPHEUS: OrpheusSettings,
    EngineType.QWEN3_TTS: QwenSettings,
    EngineType.OMNIVOICE: OmniVoiceSettings,
    EngineType.MOSS_TTS: MossTTSSettings,
    EngineType.DIA: DIASettings,
    EngineType.CSM: CSMSettings,
    EngineType.CHATTERBOX: ChatterboxSettings,
    EngineType.STYLE_TTS2: StyleTTS2Settings,
    EngineType.F5: F5Settings,
    EngineType.DMOSPEECH2: DMSpeech2Settings,
    EngineType.VIBE: VibeSettings,
}

# Mapping of engine types to their help widgets
HELP_WIDGETS: Dict[EngineType, Type] = {
    EngineType.FISH_SPEECH: FishHelp,
    EngineType.GPT_SOVITS: GPTSoVITSHelp,
    EngineType.LLASA: LlasaHelp,
    EngineType.HIGGS: HiggsHelp,
    EngineType.SPARK: SparkHelp,
    EngineType.ORPHEUS: OrpheusHelp,
    EngineType.QWEN3_TTS: QwenHelp,
    EngineType.OMNIVOICE: OmniVoiceHelp,
    EngineType.MOSS_TTS: MossTTSHelp,
    EngineType.DIA: DIAHelp,
    EngineType.CSM: CSMHelp,
    EngineType.CHATTERBOX: ChatterboxHelp,
    EngineType.STYLE_TTS2: StyleTTS2Help,
    EngineType.F5: F5Help,
    EngineType.DMOSPEECH2: DMSpeech2Help,
    EngineType.VIBE: VibeHelp,
}
