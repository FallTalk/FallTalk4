import logging
import os
import shutil

import PySide6
from PySide6.QtCore import QMetaObject, Qt, Q_ARG

from src.utils.huggingface_utils import (
    downloadXTTS, downloadRVC, downloadGPTSoVITS, downloadStyleTTS2, downloadDIA,
    downloadFish, downloadF5, downloadLlasa, downloadOrpheus, download_rvc_models
)

logger = logging.getLogger('falltalk')
logger.setLevel(logging.DEBUG)


def load_model(parent, character=None, rvc=None, display_name=None, base_model=False):
    print(f"load_model {parent} {character}")
    try:
        if not character.startswith("custom_"):
            if rvc is not None:
                download_rvc_models(character, rvc)
                downloadRVC(parent)

        if parent.tts_engine is not None:
            if character is not None:
                parent.reference_time_label.setText("00:00")
                parent.character_label.setText(f"{display_name}")
            else:
                parent.reference_time_label.setText("00:00")
                parent.character_label.setText(f"Base Model")

            parent.tts_engine.setup(character, rvc is not None, base_model)
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


def generic_engine_loader(parent, engine_class, download_func, engine_name, api=False):
    try:
        download_func(parent)
        downloadRVC(parent)
        parent.tts_engine = engine_class()
        print(f"{engine_name} Loaded")
        load_whisper(parent)
        print(f"Done Loading")
        if not api:
            QMetaObject.invokeMethod(parent, "after_engine_load", Qt.QueuedConnection, 
                                   Q_ARG(PySide6.QtCore.QObject, parent),
                                   Q_ARG(str, engine_name))
            QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, 
                                   Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, 
                               Q_ARG(PySide6.QtCore.QObject, parent), 
                               Q_ARG(str, "Unable to Load Engine"), 
                               Q_ARG(str, "An Error Occurred while loading the engine. Please check your logs and report the issue if needed"))


def load_xtts(parent):
    from src.tts_engines.xtts_engine import XTTS_Engine
    generic_engine_loader(parent, XTTS_Engine, downloadXTTS, "XTTS")


def load_whisper(parent, attempt=0):
    try:
        if parent.transcription_engine is None:
            from src.tts_engines.whisper_engine import Whisper_Engine
            parent.transcription_engine = Whisper_Engine()
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


def load_gpt_sovits(parent):
    from src.tts_engines.gpt_sovits_engine import GPT_SoVITS_Engine
    generic_engine_loader(parent, GPT_SoVITS_Engine, downloadGPTSoVITS, "GPT_SoVITS")


def load_dia(parent):
    from src.tts_engines.dia_engine import DIA_Engine
    generic_engine_loader(parent, DIA_Engine, downloadDIA, "DIA")


def load_rvc(parent, api=False):
    from src.tts_engines.rvc_engine import RVC_Engine
    generic_engine_loader(parent, RVC_Engine, downloadRVC, "RVC", api)


def load_fish(parent):
    from src.tts_engines.fish_engine import FishSpeechEngine
    generic_engine_loader(parent, FishSpeechEngine, downloadFish, "Fish")


def load_f5(parent):
    from src.tts_engines.f5_engine import F5Engine
    generic_engine_loader(parent, F5Engine, downloadF5, "F5")


def load_llasa(parent):
    from src.tts_engines.llasa_engine import LlasaEngine
    generic_engine_loader(parent, LlasaEngine, downloadLlasa, "Llasa")


def load_orpheus(parent):
    from src.tts_engines.orpheus_engine import OrpheusEngine
    generic_engine_loader(parent, OrpheusEngine, downloadOrpheus, "Orpheus")


def load_style_tts2(parent):
    from src.tts_engines.style_tts_engine import StyleTTS2_Engine
    generic_engine_loader(parent, StyleTTS2_Engine, downloadStyleTTS2, "StyleTTS2")


def load_upscaler(parent, api=False):
    try:
        if parent.upscale_engine is None:
            from src.tts_engines.upscale_engine import UpscaleEngine
            parent.upscale_engine = UpscaleEngine(parent)
            print("Upscaler Loaded")
        if not api:
            QMetaObject.invokeMethod(parent, "after_upscale", Qt.QueuedConnection, 
                                   Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, 
                               Q_ARG(PySide6.QtCore.QObject, parent), 
                               Q_ARG(str, "Unable to Load Upscaler"), 
                               Q_ARG(str, "An Error Occurred while loading the upscaler. Please check your logs and report the issue if needed"))


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