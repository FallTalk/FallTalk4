from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppCallbacks

import logging
import os
import shutil

from src.enums.engine_type import EngineType
from src.config.config import cfg
from src.utils.filesystem_utils import get_app_root, check_files_in_directory

from src.utils.huggingface_utils import (
    downloadXTTS, downloadRVC, downloadGPTSoVITS, downloadStyleTTS2, downloadDIA, downloadSpark,
    downloadFish, downloadF5, downloadLlasa, downloadOrpheus, download_rvc_models, downloadAPBWE, downloadCSM,
    downloadHiggs, downloadChatterbox, downloadDMSpeech2, downloadVibe, downloadQwen, downloadOmniVoice,
    downloadMossTTS, download_models
)

logger = logging.getLogger('falltalk')
logger.setLevel(logging.DEBUG)


def load_model(parent: 'AppCallbacks', character=None, rvc=None, display_name=None, base_model=False, model_engine_version=None):
    logger.info(f"load_model character={character} display_name={display_name} base_model={base_model}")
    try:
        state_ref = getattr(parent, "state_ref", None)
        model_registry = getattr(state_ref, "models", None) if state_ref is not None else None
        current_engine = state_ref.tts_engine.engine_type.value if state_ref is not None and state_ref.tts_engine is not None else cfg.get(cfg.engine)
        auto_downloaded = False

        if (
            not base_model
            and character is not None
            and not character.startswith("custom_")
            and isinstance(model_registry, dict)
        ):
            character_models = model_registry.get(character)
            if isinstance(character_models, dict):
                model_info = character_models.get(current_engine)
                if isinstance(model_info, dict):
                    engine_type = EngineType(current_engine)
                    engine_version = model_engine_version or model_info.get('engine_version', '1')
                    if model_info.get('is_shared', False) and model_info.get('shared_model_name'):
                        model_dir = os.path.join(
                            get_app_root(),
                            "models",
                            "shared",
                            engine_type.get_model_path(model_info['shared_model_name'], engine_version),
                        )
                    else:
                        model_dir = os.path.join(
                            get_app_root(),
                            "models",
                            engine_type.get_model_path(character, engine_version),
                        )

                    if not check_files_in_directory(model_dir):
                        logger.info(
                            "Character model files missing for %s (%s); downloading before load",
                            character,
                            current_engine,
                        )
                        if not download_models(parent, character, model_info, rvc, api=True):
                            return
                        if not check_files_in_directory(model_dir):
                            logger.error(
                                "Character model download completed but no files were found in %s",
                                model_dir,
                            )
                            parent.on_error(
                                "Unable to Load Model",
                                f"Downloaded model files for {character} were not found in {model_dir}.",
                            )
                            return
                        auto_downloaded = True

        if not (character or "").startswith("custom_"):
            if rvc is not None and not auto_downloaded:
                download_rvc_models(parent, character, rvc)
                downloadRVC(parent)

        if parent.state_ref.tts_engine is not None:
            # Check if this is a shared model
            is_shared = False
            shared_model_name = None
            characters = None

            if not base_model and character is not None and character in parent.state_ref.models:
                engine_name = parent.state_ref.tts_engine.engine_type.value
                if engine_name in parent.state_ref.models[character]:
                    model_info = parent.state_ref.models[character][engine_name]
                    is_shared = model_info.get('is_shared', False)
                    shared_model_name = model_info.get('shared_model_name')
                    characters = model_info.get('characters', [character])

            parent.state_ref.tts_engine.setup(character, rvc is not None, base_model, model_engine_version, is_shared, shared_model_name, characters)
            parent.on_model_loaded()
        else:
            logger.error("load_model called but tts_engine is None")
            parent.on_error("Unable to Load Model", "No TTS engine is currently selected.")
    except Exception as e:
        logger.exception("Unable to load model")
        parent.on_error("Unable to Load Model", "An Error Occured while attempting to load the model. Please check your logs and report the issue if needed")
    finally:
        parent.on_done()


def get_character_model(character, models, custom_models):
    if character.startswith("custom_") and custom_models is not None:
        return custom_models[character] if character in custom_models else None
    else:
        return models[character] if character in models else None


def get_trained_character(character_model, engine_name):
    rvc = False
    trained = False
    if character_model is not None:
        if "RVC" in character_model and character_model['RVC'] is not None:
            rvc = True

        if engine_name in character_model and character_model[engine_name] is not None:
            trained = True

    return trained, rvc


def get_character_models(character, characters_data):
    for character_model in characters_data:
        if character in character_model['name'] and character == character_model['name']:
            return character_model

    return None


def generic_engine_loader(parent: 'AppCallbacks', engine_class, download_func, engine_name, api=False):
    try:
        download_func(parent)
    except Exception as e:
        logger.exception(f"Error downloading the models: {e}")
        error_message = str(e)

        # Check for Hugging Face connection timeout errors
        if "huggingface.co" in error_message and ("timeout" in error_message.lower() or "connection" in error_message.lower()):
            error_title = "Hugging Face Connection Error"
            error_detail = "Unable to connect to Hugging Face servers. Please check your internet connection and try again. If the problem persists, it might be a temporary issue with Hugging Face servers. We also offer a setting to turn off secure connections, which may help"
        else:
            error_title = "Unable to Download Models"
            error_detail = f"An error occurred while downloading the models: {error_message}"

        parent.on_error(error_title, error_detail)
        return

    try:
        downloadRVC(parent)
    except Exception as e:
        logger.exception(f"Error during downloading RVC: {e}")
        parent.on_error("Unable to Download RVC",
                       f"An error occurred while downloading RVC: {str(e)}")
        return

    try:
        downloadAPBWE(parent)
    except Exception as e:
        logger.exception(f"Error downloading the upscaling engine: {e}")
        parent.on_error("Unable to Download Upscaler",
                       f"An error occurred while downloading Upscaler: {str(e)}")
        return

    try:
        parent.state_ref.tts_engine = engine_class()
        print(f"{engine_name} Loaded")
    except Exception as e:
        logger.exception(f"Error during engine initialization: {e}")
        parent.on_error("Unable to Initialize Engine",
                       f"An error occurred while initializing the {engine_name} engine: {str(e)}")
        return

    try:
        load_whisper(parent)
    except Exception as e:
        logger.exception(f"Error during transcription download: {e}")
        parent.on_error("Unable to Load Whisper",
                       f"An error occurred while loading Whisper: {str(e)}")
        return

    try:
        load_apbwe(parent)
        print(f"Done Loading")
    except Exception as e:
        logger.exception(f"Error creating upscaler: {e}")
        parent.on_error("Unable to Load APBWE",
                       f"An error occurred while loading APBWE: {str(e)}")
        return

    if not api:
        try:
            parent.on_model_loaded()
            parent.on_continue_load()
            parent.on_done()
        except Exception as e:
            logger.exception(f"Error during UI callback: {e}")
            parent.on_error("Unable to Complete Engine Loading",
                           f"An error occurred during final loading steps: {str(e)}")
            return


def load_whisper(parent: 'AppCallbacks', attempt=0):
    try:
        if parent.state_ref.transcription_engine is None:
            from src.tts_engines.whisper_engine import Whisper_Engine
            parent.state_ref.transcription_engine = Whisper_Engine()
        if parent.state_ref.tts_engine is not None and parent.state_ref.transcription_engine is not None:
            parent.state_ref.tts_engine.whisper_engine = parent.state_ref.transcription_engine
            print("WhisperX Loaded")
    except Exception as e:
        logger.exception(f"Error: {e}")
        if attempt == 0:
            from huggingface_hub.constants import HF_HUB_CACHE
            cache_dir = HF_HUB_CACHE
            if os.path.exists(HF_HUB_CACHE):
                # shutil.rmtree(cache_dir)
                print(f"Attempting to reload Whisper")
            load_whisper(parent, 1)
        else:
            parent.on_warn("Unable to Load Whispper Engine",
                          "An Error Occured while loading whisper engine. Transcription will not work for untrained models. Please delete C:\\Users\\USERNAME\\.cache\\huggingface\\hub")

def load_gpt_sovits(parent: 'AppCallbacks') -> None:
    from src.tts_engines.gpt_sovits_engine import GPT_SoVITS_Engine
    generic_engine_loader(parent, GPT_SoVITS_Engine, downloadGPTSoVITS, EngineType.GPT_SOVITS.value)


def load_dia(parent: 'AppCallbacks'):
    from src.tts_engines.dia_engine import DIA_Engine
    generic_engine_loader(parent, DIA_Engine, downloadDIA, EngineType.DIA.value)


def load_rvc(parent: 'AppCallbacks', api=False):
    from src.tts_engines.rvc_engine import RVC_Engine
    generic_engine_loader(parent, RVC_Engine, downloadRVC, EngineType.RVC.value, api)


def load_fish(parent: 'AppCallbacks'):
    from src.tts_engines.fish_engine import FishSpeechEngine
    generic_engine_loader(parent, FishSpeechEngine, downloadFish, EngineType.FISH_SPEECH.value)


def load_f5(parent: 'AppCallbacks'):
    from src.tts_engines.f5_engine import F5Engine
    generic_engine_loader(parent, F5Engine, downloadF5, EngineType.F5.value)


def load_llasa(parent: 'AppCallbacks'):
    from src.tts_engines.llasa_engine import LlasaEngine
    generic_engine_loader(parent, LlasaEngine, downloadLlasa, EngineType.LLASA.value)


def load_orpheus(parent: 'AppCallbacks'):
    from src.tts_engines.orpheus_engine import OrpheusEngine
    generic_engine_loader(parent, OrpheusEngine, downloadOrpheus, EngineType.ORPHEUS.value)


def load_style_tts2(parent: 'AppCallbacks'):
    from src.tts_engines.style_tts_engine import StyleTTS2_Engine
    generic_engine_loader(parent, StyleTTS2_Engine, downloadStyleTTS2, EngineType.STYLE_TTS2.value)


def load_spark(parent: 'AppCallbacks'):
    from src.tts_engines.spark_engine import SparkEngine
    generic_engine_loader(parent, SparkEngine, downloadSpark, EngineType.SPARK.value)

def load_csm(parent: 'AppCallbacks'):
    from src.tts_engines.csm_engine import CSMEngine
    generic_engine_loader(parent, CSMEngine, downloadCSM, EngineType.CSM.value)

def load_higgs(parent: 'AppCallbacks'):
    from src.tts_engines.higgs_tts_engine import HiggsTtsEngine
    generic_engine_loader(parent, HiggsTtsEngine, downloadHiggs, EngineType.HIGGS.value)

def load_chatterbox(parent: 'AppCallbacks'):
    from src.tts_engines.chatterbox_engine import ChatterboxEngine
    generic_engine_loader(parent, ChatterboxEngine, downloadChatterbox, EngineType.CHATTERBOX.value)

def load_dmo_speech2(parent: 'AppCallbacks'):
    from src.tts_engines.dmo_engine import DMSpeech2Engine
    generic_engine_loader(parent, DMSpeech2Engine, downloadDMSpeech2, EngineType.DMOSPEECH2.value)

def load_vibe(parent: 'AppCallbacks'):
    from src.tts_engines.vibe_engine import VibeEngine
    generic_engine_loader(parent, VibeEngine, downloadVibe, EngineType.VIBE.value)

def load_qwen(parent: 'AppCallbacks'):
    from src.tts_engines.qwen_engine import QwenEngine
    generic_engine_loader(parent, QwenEngine, downloadQwen, EngineType.QWEN3_TTS.value)

def load_omnivoice(parent: 'AppCallbacks'):
    from src.tts_engines.omnivoice_engine import OmniVoiceEngine
    generic_engine_loader(parent, OmniVoiceEngine, downloadOmniVoice, EngineType.OMNIVOICE.value)

def load_moss_tts(parent: 'AppCallbacks'):
    from src.tts_engines.moss_tts_engine import MossTTSEngine
    generic_engine_loader(parent, MossTTSEngine, downloadMossTTS, EngineType.MOSS_TTS.value)

_ENGINE_LOADERS = None

def _get_engine_loaders():
    global _ENGINE_LOADERS
    if _ENGINE_LOADERS is None:
        _ENGINE_LOADERS = {
            EngineType.GPT_SOVITS: load_gpt_sovits,
            EngineType.STYLE_TTS2: load_style_tts2,
            EngineType.DIA: load_dia,
            EngineType.LLASA: load_llasa,
            EngineType.ORPHEUS: load_orpheus,
            EngineType.FISH_SPEECH: load_fish,
            EngineType.F5: load_f5,
            EngineType.RVC: load_rvc,
            EngineType.SPARK: load_spark,
            EngineType.CSM: load_csm,
            EngineType.HIGGS: load_higgs,
            EngineType.CHATTERBOX: load_chatterbox,
            EngineType.DMOSPEECH2: load_dmo_speech2,
            EngineType.VIBE: load_vibe,
            EngineType.QWEN3_TTS: load_qwen,
            EngineType.OMNIVOICE: load_omnivoice,
            EngineType.MOSS_TTS: load_moss_tts,
        }
    return _ENGINE_LOADERS


def get_engine_loader(engine_type: EngineType):
    """Return the loader function for a given engine type, or None."""
    return _get_engine_loaders().get(engine_type)


def load_apbwe(parent: 'AppCallbacks'):
    if parent.state_ref.apbwe_engine is None:
        from src.tts_engines.apbwe_engine import APBWE_SR
        parent.state_ref.apbwe_engine = APBWE_SR()
    if parent.state_ref.tts_engine is not None and parent.state_ref.apbwe_engine is not None:
        parent.state_ref.tts_engine.apbwe_engine = parent.state_ref.apbwe_engine
        print("apbwe Loaded")


def load_upscaler(parent: 'AppCallbacks', api=False):
    if parent.state_ref.upscale_engine is None:
        from src.tts_engines.upscale_engine import UpscaleEngine
        parent.state_ref.upscale_engine = UpscaleEngine(parent)
        print("Upscaler Loaded")
    if not api:
        parent.on_done()


def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
