#!/usr/bin/env python3
"""FallTalk — imgui_bundle entry point."""
import ctypes
import logging
import os
import subprocess
import sys
import traceback


def hide_console():
    whnd = ctypes.windll.kernel32.GetConsoleWindow()
    if whnd != 0:
        ctypes.windll.user32.ShowWindow(whnd, 0)


if __name__ == '__main__':
    from src.utils.logging_utils import setup_logging

    logger = None
    try:
        # Configure the logger
        root_logger = setup_logging()
        logger = logging.getLogger('main')
    except Exception as e:
        traceback.print_exc()
        logger = logging.getLogger('main')
        logger.debug("Unable to Start FallTalk Logging", e)


    from src.config.config import cfg

    try:
        from src.utils.model_utils import seed_everything
        _seed = cfg.get(cfg.seed)
        if _seed is not None and _seed >= 0:
            seed_everything(_seed)


    except Exception as e:
        traceback.print_exc()
        logger.debug("Unable to Seed Everything", e)


    try:
        _hf_cache = cfg.get(cfg.huggingface_cache_dir)
        if _hf_cache is not None and _hf_cache != "Please Select a Valid Folder":
            os.environ["HF_HUB_CACHE"] = _hf_cache

        if cfg.get(cfg.disableSSLVerify) == True:
            import requests
            from huggingface_hub import configure_http_backend

            def backend_factory() -> requests.Session:
                session = requests.Session()
                session.verify = False
                return session

            configure_http_backend(backend_factory=backend_factory)

        if cfg.get(cfg.huggingface_key) is not None and cfg.get(cfg.huggingface_key) != "":
            os.environ['HF_TOKEN'] = cfg.get(cfg.huggingface_key)

        os.environ['WANDB_DISABLED'] = 'True'
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['TORCHDYNAMO_VERBOSE'] = '1'
        os.environ["TORCHINDUCTOR_DISABLE"] = "1"
        os.environ["TORCH_COMPILE_BACKEND"] = "eager"  # fallback to eager mode
        os.environ["PYTORCH_ENABLE_DYNAMO"] = "0"
        os.environ["TRITON_DISABLE_AUTOTUNE"] = "1"
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        os.environ["USE_LIBUV"] = "0"

        import getpass
        os.environ['USER'] = getpass.getuser()
        os.environ['PHONEMIZER_ESPEAK_PATH'] = os.path.abspath(os.path.join("resource", "apps", "espeak", "espeak-ng.exe"))
        os.environ['ESPEAK_DATA_PATH'] = os.path.abspath(os.path.join("resource", "apps", "espeak", "espeak-ng-data"))
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = os.path.abspath(os.path.join("resource", "apps", "espeak", "libespeak-ng.dll"))
        os.environ['FFMPEG_BIN'] = os.path.abspath(os.path.join("ffmpeg.exe"))
        os.environ['FFMPEG_BINARY'] = os.path.abspath(os.path.join("ffmpeg.exe"))
        os.environ['NLTK_DATA'] = os.path.abspath(os.path.join("resource", "apps", "nltk_data"))

        # Add nvidia cudnn bin dir to DLL search path (needed by ctranslate2 for cuDNN 8)
        _cudnn_bin = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cudnn", "bin")
        if os.path.isdir(_cudnn_bin):
            os.add_dll_directory(_cudnn_bin)
            os.environ['PATH'] = _cudnn_bin + os.pathsep + os.environ.get('PATH', '')

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.debug('Unable to find espeak', e)

    # Register safe globals for torch.load (needed by Pyannote/whisperx/speechbrain)
    try:
        import torch
        import omegaconf
        import omegaconf.base
        import omegaconf.nodes
        import omegaconf.listconfig
        import omegaconf.dictconfig
        import collections
        import typing
        import enum

        _safe = set()
        for mod in (omegaconf, omegaconf.base, omegaconf.nodes,
                     omegaconf.listconfig, omegaconf.dictconfig):
            for name in dir(mod):
                if name.startswith('_'):
                    continue
                obj = getattr(mod, name, None)
                if isinstance(obj, type) or isinstance(obj, enum.EnumMeta):
                    _safe.add(obj)
        _safe.update([typing.Any, set, list, collections.defaultdict, dict, int,
                      torch.torch_version.TorchVersion])

        # Pyannote types needed for whisperx VAD model loading
        from pyannote.audio.core.model import Introspection
        from pyannote.audio.core.task import Specifications, Problem, Resolution
        _safe.update([Introspection, Specifications, Problem, Resolution])

        torch.serialization.add_safe_globals(list(_safe))
    except Exception:
        pass

    try:
        app_id = 'falltalk'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)

        hide_console()

        # Launch the imgui-based UI
        from src.ui_imgui.app import run
        run()

    except Exception as e:
        traceback.print_exc()
        logger.debug("Unable to Start FallTalk", e)
