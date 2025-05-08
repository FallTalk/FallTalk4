from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp


import logging
import os
import glob
import shutil
import sys

import PySide6
import requests
from PySide6.QtCore import QMetaObject, Qt, Q_ARG

import huggingface_hub

from src.config import config
from src.config.config import cfg, REPO
from src.utils.filesystem_utils import get_app_root
from src.enums.engine_type import EngineType

logger = logging.getLogger('falltalk')
logger.setLevel(logging.DEBUG)


def download_model_from_hub(character, model) -> None:
    if model is not None:
        engine_type = EngineType(model['engine'])
        version = model.get('engine_version', 1)
        
        # Validate version compatibility
        if not engine_type.is_version_supported(version):
            raise ValueError(f"Engine version {version} is not supported for {model['engine']}")
        
        # Create version-specific directory
        model_path = engine_type.get_model_path(character, version)
        os.makedirs(os.path.join("models", model_path), exist_ok=True)
        
        # Handle file paths based on version
        if version == "1":
            filename = f"{character}_v{model['version']}.{model['type']}"
            files_to_delete = glob.glob(os.path.join("models", model_path, f"{character}*.{model['type']}"))
        else:
            filename = f"{character}_v{model['version']}.{model['type']}"
            files_to_delete = glob.glob(os.path.join("models", model_path, f"{character}*.{model['type']}"))
        
        # Clean up old files
        for file_path in files_to_delete:
            if file_path != os.path.join("models", model_path, filename):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.exception("Unable to delete old model")

        # Download the model file
        huggingface_hub.hf_hub_download(REPO, f"models/{model_path.replace(os.sep, '/')}/{filename}", local_dir=get_app_root())


def downloadXTTS(parent: FallTalkApp) -> None:
    try:
        os.makedirs(os.path.join(get_app_root(), "models/XTTSv2"), exist_ok=True)
        huggingface_hub.hf_hub_download(REPO, "models/XTTSv2/model.pth", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/XTTSv2/config.json", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/XTTSv2/vocab.json", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/XTTSv2/speakers_xtts.pth", local_dir=get_app_root())
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


def downloadRVC(parent: FallTalkApp) -> None:
    try:
        os.makedirs(os.path.join(get_app_root(), "models/RVC"), exist_ok=True)
        huggingface_hub.hf_hub_download(REPO, "models/rvc/rmvpe.onnx", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/rvc/rmvpe.pt", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/rvc/hubert_base.pt", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/rvc/fcpe.pt", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/rvc/contentvec_base.pt", local_dir=get_app_root())
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


def downloadFish(parent: FallTalkApp) -> None:
    try:
        os.makedirs(os.path.join(get_app_root(),"models/Fish"), exist_ok=True)
        huggingface_hub.hf_hub_download(REPO,"models/Fish/1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO,"models/Fish/1.5/config.json", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO,"models/Fish/1.5/model.pth", local_dir=get_app_root())
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


def downloadGPTSoVITS(parent: FallTalkApp) -> None:
    try:
        os.makedirs(os.path.join(get_app_root(),"models/GPT_SoVITS"), exist_ok=True)
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/v2/s2G2333k.pth", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/v2/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/v4/s2Gv4.pth", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/v4/vocoder.pth", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/chinese-roberta-wwm-ext-large/pytorch_model.bin", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/chinese-roberta-wwm-ext-large/tokenizer.json", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/chinese-roberta-wwm-ext-large/config.json", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/chinese-hubert-base/pytorch_model.bin", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/chinese-hubert-base/preprocessor_config.json", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/chinese-hubert-base/config.json", local_dir=get_app_root())
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


def downloadOrpheus(parent):
    try:
        os.makedirs(os.path.join(get_app_root(),"models/Orpheus"), exist_ok=True)
        # snapshot_download(
        #     repo_id="canopylabs/orpheus-tts-0.1-pretrained",
        #     allow_patterns=[
        #         "config.json",
        #         "*.safetensors",
        #         "*.json",
        #         "tokenizer.*"
        #     ]
        # )
        huggingface_hub.hf_hub_download(REPO, "models/Orpheus/3b-0.1/model-00001-of-00004.safetensors", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/Orpheus/3b-0.1/model-00002-of-00004.safetensors", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/Orpheus/3b-0.1/model-00003-of-00004.safetensors", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/Orpheus/3b-0.1/model-00004-of-00004.safetensors", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/Orpheus/3b-0.1/generation_config.json", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/Orpheus/3b-0.1/tokenizer_config.json", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/Orpheus/3b-0.1/special_tokens_map.json", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/Orpheus/3b-0.1/tokenizer.json", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/Orpheus/3b-0.1/model.safetensors.index.json", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/Orpheus/3b-0.1/config.json", local_dir=get_app_root())
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


def downloadF5(parent):
    try:
        os.makedirs(os.path.join(get_app_root(),"models/F5"), exist_ok=True)
        huggingface_hub.hf_hub_download(REPO, "models/F5/F5TTS_v1_Base/model_1250000.safetensors", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/F5/F5TTS_v1_Base/vocab.txt", local_dir=get_app_root())
        huggingface_hub.hf_hub_download("charactr/vocos-mel-24khz", "config.yaml", local_dir=os.path.abspath(f"models/F5/vocos"))
        huggingface_hub.hf_hub_download("charactr/vocos-mel-24khz", "pytorch_model.bin", local_dir=os.path.abspath(f"models/F5/vocos"))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


def downloadLlasa(parent):
    try:
        os.makedirs(os.path.join(get_app_root(),"models/Llasa/3B"), exist_ok=True)
        os.makedirs(os.path.join(get_app_root(),"models/Llasa/1B"), exist_ok=True)
        os.makedirs(os.path.join(get_app_root(),"models/Llasa/8B"), exist_ok=True)
        os.makedirs(os.path.join(get_app_root(),"models/Llasa/xcodec2"), exist_ok=True)

        if(cfg.get(cfg.llasa_mode) == '1B'):
            huggingface_hub.snapshot_download("HKUSTAudio/Llasa-1B", local_dir="models/Llasa/1B")

        if(cfg.get(cfg.llasa_mode) == '3B'):
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/3B/config.json", local_dir=get_app_root())
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/3B/generation_config.json", local_dir=get_app_root())
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/3B/model.safetensors.index.json", local_dir=get_app_root())
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/3B/model-00001-of-00002.safetensors", local_dir=get_app_root())
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/3B/model-00002-of-00002.safetensors", local_dir=get_app_root())
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/3B/special_tokens_map.json", local_dir=get_app_root())
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/3B/tokenizer.json", local_dir=get_app_root())
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/3B/tokenizer_config.json", local_dir=get_app_root())

        if(cfg.get(cfg.llasa_mode) == '8B'):
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/8B/config.json", local_dir=get_app_root())
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/8B/generation_config.json", local_dir=get_app_root())
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/8B/model.safetensors.index.json", local_dir=get_app_root())
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/8B/model-00001-of-00004.safetensors", local_dir=get_app_root())
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/8B/model-00002-of-00004.safetensors", local_dir=get_app_root())
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/8B/model-00003-of-00004.safetensors", local_dir=get_app_root())
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/8B/model-00004-of-00004.safetensors", local_dir=get_app_root())
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/8B/special_tokens_map.json", local_dir=get_app_root())
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/8B/tokenizer.json", local_dir=get_app_root())
            huggingface_hub.hf_hub_download(REPO, "models/Llasa/8B/tokenizer_config.json", local_dir=get_app_root())

        huggingface_hub.hf_hub_download(REPO, "models/Llasa/xcodec2/config.json", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/Llasa/xcodec2/model.safetensors", local_dir=get_app_root())

    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


def downloadDIA(parent):
    try:
        os.makedirs(os.path.join(get_app_root(),"models/DIA"), exist_ok=True)
        huggingface_hub.hf_hub_download(REPO, "models/DIA/0.1/dia-v0_1.pth", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/DIA/0.1/config.json", local_dir=get_app_root())
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


def downloadSpark(parent):
    try:
        os.makedirs(os.path.join(get_app_root(),"models/Spark"), exist_ok=True)
        huggingface_hub.hf_hub_download(REPO, "models/Spark/0.5B/LLM/config.json", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/Spark/0.5B/LLM/model.safetensors", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/Spark/0.5B/BiCodec/config.yml", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/Spark/0.5B/BiCodec/model.safetensors", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/Spark/0.5B/wav2vec2-large-xlsr-53/pytorch_model.bin", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/Spark/0.5B/wav2vec2-large-xlsr-53/config.json", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/Spark/0.5B/wav2vec2-large-xlsr-53/preprocessor_config.json", local_dir=get_app_root())

    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))




def downloadStyleTTS2(parent):
    try:
        os.makedirs(os.path.join(get_app_root(),"models/StyleTTS2"), exist_ok=True)
        huggingface_hub.hf_hub_download(REPO, "models/StyleTTS2/ASR/config.yml", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/StyleTTS2/ASR/epoch_00080.pth", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/StyleTTS2/JDC/bst.t7", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/StyleTTS2/PLBERT/config.yml", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/StyleTTS2/PLBERT/step_1000000.t7", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/StyleTTS2/Models/Vokan/epoch_2nd_00012.pth", local_dir=get_app_root())
        huggingface_hub.hf_hub_download(REPO, "models/StyleTTS2/Models/Vokan/config.yml", local_dir=get_app_root())
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


def downloadBaseModels(parent):
    try:
        downloadXTTS(parent)
        downloadRVC(parent)
        downloadGPTSoVITS(parent)
        downloadStyleTTS2(parent)
        downloadDIA(parent)
        QMetaObject.invokeMethod(parent, "afterDownload", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


def download_rvc_models(character, rvc):
    if rvc:
        download_model_from_hub(character, rvc)
        rvc['type'] = 'index'
        download_model_from_hub(character, rvc)


def download_models(parent, character, model, rvc, api=False):
    try:
        # Validate engine version before downloading
        engine_type = EngineType(model['engine'])
        version = model.get('engine_version', 1)
        
        if not engine_type.is_version_supported(version):
            raise ValueError(f"Engine version {version} is not supported for {model['engine']}")
        
        download_model_from_hub(character, model)
        download_rvc_models(character, rvc)
        
        if model['engine'] == "GPT_SoVITS":
            model['type'] = "ckpt"
            download_model_from_hub(character, model)

        if not api:
            QMetaObject.invokeMethod(parent, "afterModelDownload", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception("Unable to download model")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Model"), Q_ARG(str, "An Error Occured while downloading the model from huggingface. Please check your logs and report the issue if needed"))

# Additional functions that might be needed for backward compatibility
def get_latest_release():
    try:
        response = requests.get(config.RELEASE_URL)
        if response.status_code == 200:
            return response.url.split('/')[-1]
        else:
            logger.exception(f"Failed to fetch latest release: {response.status_code}")
            return config.VERSION
    except Exception as e:
        logger.exception(f"Failed to fetch latest release: {e}")
        return config.VERSION


def get_model_diff(old_json, new_json):
    result = []
    if old_json is not None and new_json is not None:
        old_characters = old_json.get('characters', [])
        new_characters = new_json.get('characters', [])

        old_entries = {entry['name']: entry for entry in old_characters}
        new_entries = {entry['name']: entry for entry in new_characters}

        for name in new_entries:
            if name not in old_entries:
                new_entry = new_entries[name]
                new_models = [key for key in new_entry if key not in ['name', 'display_name']]
                result.append(f"Added: {new_entry['display_name']}\n    -Engine: {', '.join(new_models)}")
                continue

            old_entry = old_entries[name]
            new_entry = new_entries[name]

            model_changes = []
            for key in new_entry:
                if key not in old_entry:
                    model_changes.append(f" -Added: {key}")
                elif isinstance(new_entry[key], dict):
                    old_version = old_entry[key].get('version')
                    new_version = new_entry[key].get('version')
                    if old_version != new_version:
                        model_changes.append(f" -Updated: {key} version {new_version}")

            if model_changes:
                changes = "\n ".join(model_changes)
                result.append(f"Changed: {new_entry['display_name']}\n      {changes}")

    if not result:
        return None

    return "\n \n".join(result)
