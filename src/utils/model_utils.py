from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

import logging
import os
import shutil

import PySide6
from PySide6.QtCore import QMetaObject, Qt, Q_ARG

from src.enums.engine_type import EngineType

from src.utils.huggingface_utils import (
    downloadXTTS, downloadRVC, downloadGPTSoVITS, downloadStyleTTS2, downloadDIA, downloadSpark,
    downloadFish, downloadF5, downloadLlasa, downloadOrpheus, download_rvc_models, downloadAPBWE, downloadCSM,
    downloadHiggs, downloadChatterbox, downloadDMSpeech2
)

logger = logging.getLogger('falltalk')
logger.setLevel(logging.DEBUG)


def load_model(parent: 'FallTalkApp', character=None, rvc=None, display_name=None, base_model=False, model_engine_version=None):
    print(f"load_model {parent} {character}")
    try:
        if not character.startswith("custom_"):
            if rvc is not None:
                download_rvc_models(parent, character, rvc)
                downloadRVC(parent)

        if parent.tts_engine is not None:
            if character is not None:
                parent.character_label.setText(f"{display_name}")
            else:
                parent.character_label.setText(f"Base Model")

            # Reset reference audio
            if hasattr(parent, 'references_widget'):
                parent.references_widget.clear()
            else:
                parent.reference_label.setText(parent.tr("Default Reference"))
                parent.reference_time_label.setVisible(False)

            # Check if this is a shared model
            is_shared = False
            shared_model_name = None
            characters = None

            if not base_model and character in parent.models:
                engine_name = parent.tts_engine.engine_type.value
                if engine_name in parent.models[character]:
                    model_info = parent.models[character][engine_name]
                    is_shared = model_info.get('is_shared', False)
                    shared_model_name = model_info.get('shared_model_name')
                    characters = model_info.get('characters', [character])

            parent.tts_engine.setup(character, rvc is not None, base_model, model_engine_version, is_shared, shared_model_name, characters)
            QMetaObject.invokeMethod(parent, "afterModelLoader", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception("Unable to load model")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Model"), Q_ARG(str, "An Error Occured while attempting to load the model. Please check your logs and report the issue if needed"))


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


def generic_engine_loader(parent: 'FallTalkApp', engine_class, download_func, engine_name, api=False):
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

        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection,
                               Q_ARG(PySide6.QtCore.QObject, parent),
                               Q_ARG(str, error_title),
                               Q_ARG(str, error_detail))
        return

    try:
        downloadRVC(parent)
    except Exception as e:
        logger.exception(f"Error during downloading RVC: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection,
                               Q_ARG(PySide6.QtCore.QObject, parent),
                               Q_ARG(str, "Unable to Download RVC"),
                               Q_ARG(str, f"An error occurred while downloading RVC: {str(e)}"))
        return

    try:
        downloadAPBWE(parent)
    except Exception as e:
        logger.exception(f"Error downloading the upscaling engine: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection,
                               Q_ARG(PySide6.QtCore.QObject, parent),
                               Q_ARG(str, "Unable to Download Upscaler"),
                               Q_ARG(str, f"An error occurred while downloading Upscaler: {str(e)}"))
        return

    try:
        parent.tts_engine = engine_class()
        print(f"{engine_name} Loaded")
    except Exception as e:
        logger.exception(f"Error during engine initialization: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection,
                               Q_ARG(PySide6.QtCore.QObject, parent),
                               Q_ARG(str, "Unable to Initialize Engine"),
                               Q_ARG(str, f"An error occurred while initializing the {engine_name} engine: {str(e)}"))
        return

    try:
        load_whisper(parent)
    except Exception as e:
        logger.exception(f"Error during transcription download: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection,
                               Q_ARG(PySide6.QtCore.QObject, parent),
                               Q_ARG(str, "Unable to Load Whisper"),
                               Q_ARG(str, f"An error occurred while loading Whisper: {str(e)}"))
        return

    try:
        load_apbwe(parent)
        print(f"Done Loading")
    except Exception as e:
        logger.exception(f"Error creating upscaler: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection,
                               Q_ARG(PySide6.QtCore.QObject, parent),
                               Q_ARG(str, "Unable to Load APBWE"),
                               Q_ARG(str, f"An error occurred while loading APBWE: {str(e)}"))
        return

    if not api:
        try:
            QMetaObject.invokeMethod(parent, "after_engine_load", Qt.QueuedConnection,
                                   Q_ARG(PySide6.QtCore.QObject, parent),
                                   Q_ARG(str, engine_name))
            QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection,
                                   Q_ARG(PySide6.QtCore.QObject, parent))
        except Exception as e:
            logger.exception(f"Error during UI callback: {e}")
            QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection,
                                   Q_ARG(PySide6.QtCore.QObject, parent),
                                   Q_ARG(str, "Unable to Complete Engine Loading"),
                                   Q_ARG(str, f"An error occurred during final loading steps: {str(e)}"))
            return





def load_whisper(parent: 'FallTalkApp', attempt=0):
    try:
        if parent.transcription_engine is None:
            from src.tts_engines.whisper_engine import Whisper_Engine
            parent.transcription_engine = Whisper_Engine()
        if parent.tts_engine is not None and parent.transcription_engine is not None:
            parent.tts_engine.whisper_engine = parent.transcription_engine
            print("WhisperX Loaded")
    except Exception as e:
        logger.exception(f"Error: {e}")
        if attempt == 0:
            from huggingface_hub.constants import HF_HUB_CACHE
            cache_dir = HF_HUB_CACHE
            if os.path.exists(HF_HUB_CACHE):
                shutil.rmtree(cache_dir)
                print(f"Attempting to delete HuggingFace cache directory: {cache_dir}")
            load_whisper(parent, 1)
        else:
            QMetaObject.invokeMethod(parent, "onWarn", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Whispper Engine"),
                                     Q_ARG(str, "An Error Occured while loading whisper engine. Transcription will not work for untrained models. Please delete C:\\Users\\USERNAME\\.cache\\huggingface\\hub"))


def load_xtts(parent: 'FallTalkApp'):
    from src.tts_engines.xtts_engine import XTTS_Engine
    generic_engine_loader(parent, XTTS_Engine, downloadXTTS, EngineType.XTTS_V2.value)


def load_gpt_sovits(parent: 'FallTalkApp') -> None:
    from src.tts_engines.gpt_sovits_engine import GPT_SoVITS_Engine
    generic_engine_loader(parent, GPT_SoVITS_Engine, downloadGPTSoVITS, EngineType.GPT_SOVITS.value)


def load_dia(parent: 'FallTalkApp'):
    from src.tts_engines.dia_engine import DIA_Engine
    generic_engine_loader(parent, DIA_Engine, downloadDIA, EngineType.DIA.value)


def load_rvc(parent: 'FallTalkApp', api=False):
    from src.tts_engines.rvc_engine import RVC_Engine
    generic_engine_loader(parent, RVC_Engine, downloadRVC, EngineType.RVC.value, api)


def load_fish(parent: 'FallTalkApp'):
    from src.tts_engines.fish_engine import FishSpeechEngine
    generic_engine_loader(parent, FishSpeechEngine, downloadFish, EngineType.FISH_SPEECH.value)


def load_f5(parent: 'FallTalkApp'):
    from src.tts_engines.f5_engine import F5Engine
    generic_engine_loader(parent, F5Engine, downloadF5, EngineType.F5.value)


def load_llasa(parent: 'FallTalkApp'):
    from src.tts_engines.llasa_engine import LlasaEngine
    generic_engine_loader(parent, LlasaEngine, downloadLlasa, EngineType.LLASA.value)


def load_orpheus(parent: 'FallTalkApp'):
    from src.tts_engines.orpheus_engine import OrpheusEngine
    generic_engine_loader(parent, OrpheusEngine, downloadOrpheus, EngineType.ORPHEUS.value)


def load_style_tts2(parent: 'FallTalkApp'):
    from src.tts_engines.style_tts_engine import StyleTTS2_Engine
    generic_engine_loader(parent, StyleTTS2_Engine, downloadStyleTTS2, EngineType.STYLE_TTS2.value)


def load_spark(parent: 'FallTalkApp'):
    from src.tts_engines.spark_engine import SparkEngine
    generic_engine_loader(parent, SparkEngine, downloadSpark, EngineType.SPARK.value)

def load_csm(parent: 'FallTalkApp'):
    from src.tts_engines.csm_engine import CSMEngine
    generic_engine_loader(parent, CSMEngine, downloadCSM, EngineType.CSM.value)

def load_higgs(parent: 'FallTalkApp'):
    from src.tts_engines.higgs_tts_engine import HiggsTtsEngine
    generic_engine_loader(parent, HiggsTtsEngine, downloadHiggs, EngineType.HIGGS.value)

def load_chatterbox(parent: 'FallTalkApp'):
    from src.tts_engines.chatterbox_engine import ChatterboxEngine
    generic_engine_loader(parent, ChatterboxEngine, downloadChatterbox, EngineType.CHATTERBOX.value)

def load_dmo_speech2(parent: 'FallTalkApp'):
    from src.tts_engines.dmo_engine import DMSpeech2Engine
    generic_engine_loader(parent, DMSpeech2Engine, downloadDMSpeech2, EngineType.DMOSPEECH2.value)

def load_apbwe(parent: 'FallTalkApp'):
    if parent.apbwe_engine is None:
        from src.tts_engines.apbwe_engine import APBWE_SR
        parent.apbwe_engine = APBWE_SR()
    if parent.tts_engine is not None and parent.apbwe_engine is not None:
        parent.tts_engine.apbwe_engine = parent.apbwe_engine
        print("apbwe Loaded")


def load_upscaler(parent: 'FallTalkApp', api=False):
    if parent.upscale_engine is None:
        from src.tts_engines.upscale_engine import UpscaleEngine
        parent.upscale_engine = UpscaleEngine(parent)
        print("Upscaler Loaded")
    if not api:
        QMetaObject.invokeMethod(parent, "after_upscale", Qt.QueuedConnection,
                               Q_ARG(PySide6.QtCore.QObject, parent))


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
