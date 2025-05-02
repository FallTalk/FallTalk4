import logging
import os
import shutil
from PySide6.QtCore import QMetaObject, Qt, Q_ARG
import PySide6

from src.config.config import cfg
from src.utils.huggingface_utils import (
    downloadXTTS, downloadRVC, downloadGPTSoVITS, downloadStyleTTS2, downloadDIA,
    downloadFish, downloadF5, downloadLlasa, downloadOrpheus, download_models, download_rvc_models
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


def load_xtts(parent):
    try:
        downloadXTTS(parent)
        downloadRVC(parent)
        from src.tts_engines.xtts_engine import XTTS_Engine
        parent.tts_engine = XTTS_Engine()
        print("XTTS Loaded")
        load_whisper(parent)
        QMetaObject.invokeMethod(parent, "afterXtts", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
        QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Engine"), Q_ARG(str, "An Error Occured while loading the engine. Please check your logs and report the issue if needed"))


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
    try:
        downloadGPTSoVITS(parent)
        downloadRVC(parent)
        from src.tts_engines.gpt_sovits_engine import GPT_SoVITS_Engine
        from src.tts_engines.whisper_engine import Whisper_Engine
        parent.tts_engine = GPT_SoVITS_Engine()
        print("GPT_SoVITS Loaded")
        load_whisper(parent)
        QMetaObject.invokeMethod(parent, "afterGPT_SoVITS", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
        QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Engine"), Q_ARG(str, "An Error Occured while loading the engine. Please check your logs and report the issue if needed"))


def load_dia(parent):
    try:
        downloadDIA(parent)
        downloadRVC(parent)
        from src.tts_engines.dia_engine import DIA_Engine
        from src.tts_engines.whisper_engine import Whisper_Engine
        parent.tts_engine = DIA_Engine()
        print("DIA Loaded")
        load_whisper(parent)
        QMetaObject.invokeMethod(parent, "afterDIA", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
        QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Engine"), Q_ARG(str, "An Error Occured while loading the engine. Please check your logs and report the issue if needed"))


def load_rvc(parent, api=False):
    try:
        downloadRVC(parent)
        from src.tts_engines.rvc_engine import RVC_Engine
        parent.tts_engine = RVC_Engine()
        print("RVC Loaded")
        load_whisper(parent)
        if not api:
            QMetaObject.invokeMethod(parent, "afterRVC", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
            QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Engine"), Q_ARG(str, "An Error Occured while loading the engine. Please check your logs and report the issue if needed"))


def load_fish(parent):
    try:
        downloadFish(parent)
        downloadRVC(parent)
        print("Fish downloaded")
        from src.tts_engines.fish_engine import FishSpeechEngine
        parent.tts_engine = FishSpeechEngine()
        print("Fish Loaded")
        load_whisper(parent)
        QMetaObject.invokeMethod(parent, "afterFish", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
        QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Engine"), Q_ARG(str, "An Error Occured while loading the engine. Please check your logs and report the issue if needed"))


def load_f5(parent):
    try:
        downloadF5(parent)
        downloadRVC(parent)
        print("F5 downloaded")
        from src.tts_engines.f5_engine import F5Engine
        parent.tts_engine = F5Engine()
        print("F5 Loaded")
        load_whisper(parent)
        QMetaObject.invokeMethod(parent, "afterF5", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
        QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Engine"), Q_ARG(str, "An Error Occured while loading the engine. Please check your logs and report the issue if needed"))


def load_llasa(parent):
    try:
        downloadLlasa(parent)
        downloadRVC(parent)
        print("Llasa downloaded")
        from src.tts_engines.llasa_engine import LlasaEngine
        parent.tts_engine = LlasaEngine()
        print("Llasa Loaded")
        load_whisper(parent)
        QMetaObject.invokeMethod(parent, "afterLlasa", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
        QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Engine"), Q_ARG(str, "An Error Occured while loading the engine. Please check your logs and report the issue if needed"))


def load_orpheus(parent):
    try:
        downloadOrpheus(parent)
        downloadRVC(parent)
        print("Orpheus downloaded")
        from src.tts_engines.llasa_engine import LlasaEngine
        parent.tts_engine = LlasaEngine()
        print("Orpheus Loaded")
        load_whisper(parent)
        QMetaObject.invokeMethod(parent, "afterOrpheus", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
        QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Engine"), Q_ARG(str, "An Error Occured while loading the engine. Please check your logs and report the issue if needed"))


def load_style_tts2(parent):
    try:
        downloadStyleTTS2(parent)
        downloadRVC(parent)
        print("StyleTTS2 downloaded")
        from src.tts_engines.style_tts_engine import StyleTTS2_Engine
        parent.tts_engine = StyleTTS2_Engine()
        print("StyleTTS2 Loaded")
        load_whisper(parent)
        QMetaObject.invokeMethod(parent, "afterStyleTTS2", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
        QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Engine"), Q_ARG(str, "An Error Occured while loading the engine. Please check your logs and report the issue if needed"))


def load_upscaler(parent, api=False):
    try:
        if parent.upscale_engine is None:
            from src.tts_engines.upscale_engine import UpscaleEngine
            parent.upscale_engine = UpscaleEngine(parent)
            print("Upscaler Loaded")
        if not api:
            QMetaObject.invokeMethod(parent, "after_upscale", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Upscaler"), Q_ARG(str, "An Error Occured while loading the upscaler. Please check your logs and report the issue if needed"))


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