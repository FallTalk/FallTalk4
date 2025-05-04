import sys
import os
import ctypes
import subprocess
import traceback

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

import src.api.falltalkapi as falltalkapi
from src.FallTalk import ModelApp
from src.utils.model_utils import seed_everything
from src.config.config import cfg
from src.utils.logging_utils import setup_logging
from src.utils import logging_utils

def hide_console():
    whnd = ctypes.windll.kernel32.GetConsoleWindow()
    if whnd != 0:
        ctypes.windll.user32.ShowWindow(whnd, 0)


if __name__ == '__main__':

    try:
        # Configure the logger
        root_logger = setup_logging()
        logger = logging_utils.logger
        seed_everything(cfg.get(cfg.seed))

        if cfg.get(cfg.dpiScale) != "Auto":
            os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
            os.environ["QT_SCALE_FACTOR"] = str(cfg.get(cfg.dpiScale))

        application = QApplication(sys.argv)
    except Exception as e:
        traceback.print_exc()
        print("Unable to Start FallTalk", e)

    try:

        if cfg.get(cfg.huggingface_cache_dir) != "Please Select a Valid Folder":
            os.environ["HF_HUB_CACHE"] = cfg.get(cfg.huggingface_cache_dir)

        if cfg.get(cfg.disableSSLVerify) == True:
            import requests
            from huggingface_hub import configure_http_backend

            def backend_factory() -> requests.Session:
                session = requests.Session()
                session.verify = False
                return session

            configure_http_backend(backend_factory=backend_factory)

        if cfg.get(cfg.huggingface_key) is not None and cfg.get(cfg.huggingface_key != ""):
            os.environ['HF_TOKEN'] = cfg.huggingface_key

        os.environ['WANDB_DISABLED'] = 'True'
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        import getpass
        os.environ['USER'] = getpass.getuser()
        os.environ['PHONEMIZER_ESPEAK_PATH'] = os.path.abspath(os.path.join("resource", "apps", "espeak", "espeak-ng.exe"))
        os.environ['ESPEAK_DATA_PATH'] = os.path.abspath(os.path.join("resource", "apps", "espeak", "espeak-ng-data"))
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = os.path.abspath(os.path.join("resource", "apps", "espeak", "libespeak-ng.dll"))
        os.environ['FFMPEG_BIN'] = os.path.abspath(os.path.join("ffmpeg.exe"))
        os.environ['FFMPEG_BINARY'] = os.path.abspath(os.path.join("ffmpeg.exe"))
        os.environ['NLTK_DATA'] = os.path.abspath(os.path.join("resource", "apps", "nltk_data"))

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        from src.utils import logging_utils
        logger = logging_utils.logger
        logger.debug('Unable to find espeak', e)

    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.Ceil)

        app_id = 'falltalk'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)

        if cfg.get(cfg.first_start):
            from qfluentwidgets import Theme
            cfg.set(cfg.themeMode, Theme.DARK)

        falltak_app = ModelApp()
        api_server = falltalkapi.FallTalkAPI(falltak_app)
        hide_console()
        application.exec()
        print(f"Shutting down")
        api_server.shutdown()
        sys.exit()
    except Exception as e:
        traceback.print_exc()
        print("Unable to Start FallTalk", e)
