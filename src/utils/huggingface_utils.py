from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

from huggingface_hub import snapshot_download

import logging
import os
import shutil
from tqdm.auto import tqdm
import PySide6
import requests
from PySide6.QtCore import QMetaObject, Qt, Q_ARG
import huggingface_hub


class FallTalkTqdm(tqdm):
    """Custom tqdm class for FallTalk that updates the UI with download progress."""

    _parent = None

    @classmethod
    def set_parent(cls, parent):
        """Set the parent FallTalkApp instance for all instances of this class."""
        cls._parent = parent

    def update(self, n=1):
        """Override update method to also update the UI."""
        super().update(n)
        if self._parent is not None:
            # Format the progress message
            if self.total is not None:
                msg = f"{self.desc}: {self.n}/{self.total} [{self.percentage:.0f}%]"
            else:
                msg = f"{self.desc}: {self.n} items"

            # Update the UI using QMetaObject to safely call from a background thread
            QMetaObject.invokeMethod(
                self._parent, 
                "update_loader", 
                Qt.QueuedConnection, 
                Q_ARG(str, msg)
            )

    @property
    def percentage(self):
        """Calculate percentage complete."""
        return 100 * self.n / self.total if self.total else 0

from src.config.config import cfg, REPO, VERSION, RELEASE_URL
from src.utils.filesystem_utils import get_app_root
from src.enums.engine_type import EngineType

logger = logging.getLogger('falltalk')
logger.setLevel(logging.DEBUG)


def download_model_from_hub(parent: 'FallTalkApp', character, model) -> None:
    if model is not None:
        # Set the parent for the tqdm class
        FallTalkTqdm.set_parent(parent)

        engine_type = EngineType(model['engine'])
        version = model.get('engine_version', 1)

        # Validate version compatibility
        if not engine_type.is_version_supported(version):
            raise ValueError(f"Engine version {version} is not supported for {model['engine']}")

        if model.get('is_shared', False):
            model_path = os.path.join("shared", engine_type.get_model_path(model['shared_model_name'], version))
        else:
            model_path = os.path.join(engine_type.get_model_path(character, version))

        dirpath = os.path.join(get_app_root(), "models", model_path)
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        os.makedirs(dirpath, exist_ok=True)

        # Download the model file
        snapshot_download(
            repo_id=REPO,
            allow_patterns=[f"models/{model_path}/*"],
            local_dir=get_app_root(),
            local_dir_use_symlinks=False,
            tqdm_class=FallTalkTqdm,
        )


def downloadXTTS(parent: 'FallTalkApp') -> None:
    # Set the parent for the tqdm class
    FallTalkTqdm.set_parent(parent)

    os.makedirs(os.path.join(get_app_root(), "models/XTTSv2"), exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        allow_patterns=["models/XTTSv2/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
        tqdm_class=FallTalkTqdm,
    )

def downloadRVC(parent: 'FallTalkApp') -> None:
    # Set the parent for the tqdm class
    FallTalkTqdm.set_parent(parent)

    os.makedirs(os.path.join(get_app_root(), "models/RVC"), exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        allow_patterns=["models/rvc/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
        tqdm_class=FallTalkTqdm,
    )

def downloadFish(parent: 'FallTalkApp') -> None:
    # Set the parent for the tqdm class
    FallTalkTqdm.set_parent(parent)

    os.makedirs(os.path.join(get_app_root(),"models/Fish"), exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        allow_patterns=["models/Fish/s1-mini/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
        tqdm_class=FallTalkTqdm,
    )

def downloadGPTSoVITS(parent: 'FallTalkApp') -> None:
    # Set the parent for the tqdm class
    FallTalkTqdm.set_parent(parent)

    os.makedirs(os.path.join(get_app_root(),"models/GPT_SoVITS"), exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        allow_patterns=["models/GPT_SoVITS/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
        tqdm_class=FallTalkTqdm,
    )

def downloadOrpheus(parent: 'FallTalkApp'):
    # Set the parent for the tqdm class
    FallTalkTqdm.set_parent(parent)

    os.makedirs(os.path.join(get_app_root(),"models/Orpheus"), exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        allow_patterns=["models/Orpheus/3b-0.1/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
        tqdm_class=FallTalkTqdm,
    )

def downloadF5(parent: 'FallTalkApp'):
    # Set the parent for the tqdm class
    FallTalkTqdm.set_parent(parent)

    os.makedirs(os.path.join(get_app_root(),"models/F5"), exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        allow_patterns=["models/F5/F5TTS_v1_Base/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
        tqdm_class=FallTalkTqdm,
    )
    huggingface_hub.hf_hub_download("charactr/vocos-mel-24khz", "config.yaml", local_dir=os.path.abspath(f"models/F5/vocos"))
    huggingface_hub.hf_hub_download("charactr/vocos-mel-24khz", "pytorch_model.bin", local_dir=os.path.abspath(f"models/F5/vocos"))

def downloadLlasa(parent: 'FallTalkApp'):
    # Set the parent for the tqdm class
    FallTalkTqdm.set_parent(parent)

    os.makedirs(os.path.join(get_app_root(),"models/Llasa/3B"), exist_ok=True)
    os.makedirs(os.path.join(get_app_root(),"models/Llasa/1B"), exist_ok=True)
    os.makedirs(os.path.join(get_app_root(),"models/Llasa/xcodec2"), exist_ok=True)

    snapshot_download(
        repo_id=REPO,
        allow_patterns=["models/Llasa/xcodec2/*", f"models/Llasa/{cfg.get(cfg.llasa_mode)}/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
        tqdm_class=FallTalkTqdm,
    )

def downloadAPBWE(parent: 'FallTalkApp'):
    # Set the parent for the tqdm class
    FallTalkTqdm.set_parent(parent)

    os.makedirs(os.path.join(get_app_root(),"models/APBWE"), exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        allow_patterns=["models/APBWE/16kto48k/*", "models/APBWE/24kto48k/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
        tqdm_class=FallTalkTqdm,
    )

def downloadDIA(parent: 'FallTalkApp'):
    # Set the parent for the tqdm class
    FallTalkTqdm.set_parent(parent)

    os.makedirs(os.path.join(get_app_root(),"models/DIA"), exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        allow_patterns=[f"models/DIA/3b-0.1/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
        tqdm_class=FallTalkTqdm,
    )

def downloadCSM(parent: 'FallTalkApp'):
    # Set the parent for the tqdm class
    FallTalkTqdm.set_parent(parent)

    snapshot_download(
        repo_id=REPO,
        allow_patterns=[f"models/CSM/1b/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
        tqdm_class=FallTalkTqdm,
    )

def downloadSpark(parent: 'FallTalkApp'):
    # Set the parent for the tqdm class
    FallTalkTqdm.set_parent(parent)

    os.makedirs(os.path.join(get_app_root(),"models/Spark"), exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        allow_patterns=[f"models/Spark/0.5B/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
        tqdm_class=FallTalkTqdm,
    )


def downloadHiggs(parent: 'FallTalkApp'):
    # Set the parent for the tqdm class
    FallTalkTqdm.set_parent(parent)

    os.makedirs(os.path.join(get_app_root(),"models/Higgs"), exist_ok=True)
    os.makedirs(os.path.join(get_app_root(),"models/Higgs/v2"), exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        allow_patterns=[f"models/Higgs/v2/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
        tqdm_class=FallTalkTqdm,
    )

def downloadChatterbox(parent: 'FallTalkApp'):
    # Set the parent for the tqdm class
    FallTalkTqdm.set_parent(parent)

    os.makedirs(os.path.join(get_app_root(),"models/Chatterbox"), exist_ok=True)
    os.makedirs(os.path.join(get_app_root(),"models/Chatterbox/0.5B"), exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        allow_patterns=[f"models/Chatterbox/0.5B/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
        tqdm_class=FallTalkTqdm,
    )

def downloadDMSpeech2(parent: 'FallTalkApp'):
    # Set the parent for the tqdm class
    FallTalkTqdm.set_parent(parent)

    os.makedirs(os.path.join(get_app_root(),"models/DMOSpeech2"), exist_ok=True)
    os.makedirs(os.path.join(get_app_root(),"models/DMOSpeech2/v2"), exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        allow_patterns=[f"models/DMOSpeech2/v2/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
        tqdm_class=FallTalkTqdm,
    )

def downloadVibe(parent: 'FallTalkApp'):
    # Set the parent for the tqdm class
    FallTalkTqdm.set_parent(parent)

    os.makedirs(os.path.join(get_app_root(),"models/Vibe"), exist_ok=True)

    snapshot_download(
        repo_id=REPO,
        allow_patterns=[f"models/Vibe/{cfg.get(cfg.vibe_mode)}/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
        tqdm_class=FallTalkTqdm,
    )

def downloadStyleTTS2(parent: 'FallTalkApp'):
    # Set the parent for the tqdm class
    FallTalkTqdm.set_parent(parent)

    os.makedirs(os.path.join(get_app_root(),"models/StyleTTS2"), exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        allow_patterns=[f"models/StyleTTS2/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
        tqdm_class=FallTalkTqdm,
    )

def download_rvc_models(parent: 'FallTalkApp', character, rvc):
    if rvc:
        download_model_from_hub(parent, character, rvc)

def download_models(parent, character, model, rvc, api=False):
    try:
        # Validate engine version before downloading
        engine_type = EngineType(model['engine'])
        version = model.get('engine_version', 1)

        if not engine_type.is_version_supported(version):
            raise ValueError(f"Engine version {version} is not supported for {model['engine']}")

        download_model_from_hub(parent, character, model)
        download_rvc_models(parent, character, rvc)

        if not api:
            QMetaObject.invokeMethod(parent, "afterModelDownload", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception("Unable to download model")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Model"), Q_ARG(str, "An Error Occured while downloading the model from huggingface. Please check your logs and report the issue if needed"))

# Additional functions that might be needed for backward compatibility
def get_latest_release():
    try:
        response = requests.get(RELEASE_URL)
        if response.status_code == 200:
            return response.url.split('/')[-1]
        else:
            logger.exception(f"Failed to fetch latest release: {response.status_code}")
            return VERSION
    except Exception as e:
        logger.exception(f"Failed to fetch latest release: {e}")
        return VERSION


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


def download_all_models_config():
    # Note: This function doesn't have a parent parameter, so we can't update the UI
    os.makedirs(os.path.join(get_app_root(), "models"), exist_ok=True)
    os.makedirs(os.path.join(get_app_root(), "config"), exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        allow_patterns=[f"config/*"],
        local_dir=get_app_root(),
        local_dir_use_symlinks=False,
    )
