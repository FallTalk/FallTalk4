import copy
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from icons import FallTalkIcons, FallTalkStrokeIcons

# Configure the logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

os.makedirs("logs", exist_ok=True)
# Create a handler that writes log messages to a file, with rotation
handler = RotatingFileHandler('logs/falltalk.log', maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8')
handler.setLevel(logging.INFO)

# Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# Add the handler to the logger
root_logger.addHandler(handler)


class LoggerStream:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            try:
                self.logger.log(self.level, message.strip())
            except UnicodeEncodeError:
                # Fallback to a safe encoding
                safe_message = message.encode('utf-8', errors='replace').decode('utf-8')
                self.logger.log(self.level, safe_message.strip())

    def flush(self):
        pass

    def isatty(self):
        return False


stdout_logger = logging.getLogger('stdout')
stderr_logger = logging.getLogger('stderr')

# Redirect stdout and stderr to the logger
sys.stdout = LoggerStream(stdout_logger, logging.INFO)
sys.stderr = LoggerStream(stderr_logger, logging.ERROR)

import pip_system_certs.wrapt_requests

import ctypes
import getpass
import json
import shutil
import subprocess
import threading
import uuid
import webbrowser

import PySide6
import huggingface_hub
from PySide6.QtCore import Qt, QSize, Slot, QUrl, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from qfluentwidgets import FluentIcon as FIF, SplashScreen, StateToolTip, Dialog, Flyout, InfoBarIcon, InfoBar, InfoBarPosition, MessageBox, TextWrap
from qfluentwidgets import Theme, NavigationItemPosition
from packaging import version
import config
import falltalkapi
from falltalk import falltalkutils
from falltalk.Widgets import SettingsWidget, CharactersWidget, ReferencesWidget, XttsWidget, VoiceCraftWidget, FaqWidget, GPT_SoVITSWidget, StyleTTS2Widget, RVCWidget, FallTalkFluentWindow, FallTalkWidget, BulkGenerationWidget, UpscaleWidget, MusicGenWidget, AudioGenWidget
from falltalk.config import cfg, DISCLAIMER, REPO


class ModelApp(FallTalkFluentWindow):
    def __init__(self):
        super().__init__()
        self.stateTooltip = None
        self.icon_path = "resource/falltalk.png"  # Replace with the actual path to your icon file
        self.icon = QIcon(self.icon_path)
        self.splashScreen = SplashScreen(self.icon, self)
        self.splashScreen.setIconSize(QSize(128, 128))
        self.setupWindow()

        self.tts_engine = None
        self.transcription_engine = None
        self.pending_character = None
        self.pending_rvc = None
        self.pending_model = None
        self.pending_base = False
        self.characters_data = None
        self.models = None
        self.custom_models = None
        self.pending_bulk = False
        self.upscale_engine = None
        self.music_engine = None
        self.sound_fx_engine = None

        falltalkutils.clean_folder("temp/")

        self.initUI()

        if cfg.get(cfg.download_configs) or cfg.get(cfg.first_start):
            self.download_models_config()

        if cfg.get(cfg.check_for_updates):
            self.checkForRelease()

        self.load_models_config()

        self.splashScreen.finish()

        if not cfg.get(cfg.accepted_disclaimer):
            self.verify()

        self.verifyFallout()

        if cfg.get(cfg.load_engine_art_start) and not cfg.get(cfg.first_start):
            self.onEngineChange(cfg.engine)

        if cfg.get(cfg.first_start):
            self.downloadModels()
            cfg.set(cfg.first_start, False)

        if cfg.get(cfg.api_only_mode):
            self.setVisible(False)

    def checkForRelease(self):
        try:
            latest = falltalkutils.get_latest_release()
            my_version = version.parse(config.VERSION)
            latest_version = version.parse(latest)
            if latest_version.base_version > my_version.base_version:
                self.toolbar_1.addSeparator()
                self.toolbar_1.addAction(self.update_action)
                self.update_action.triggered.connect(lambda: webbrowser.open(config.RELEASE_URL))
        except Exception as e:
            falltalkutils.logger.exception(f"Failed to fetch latest release: {e}")

    def setupWindow(self):
        self.setWindowTitle(f'FallTalk - {config.VERSION}')
        self.setWindowIcon(self.icon)
        desktop = QApplication.screens()[0].availableGeometry()
        # Set size
        self.setMinimumSize(1280, 900)
        self.resize(1280, 900)
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)

        QApplication.processEvents()
        self.show()
        QApplication.processEvents()

    def verifyFallout(self):
        if cfg.get(cfg.fallout_4_directory_check):
            if cfg.get(cfg.fallout_4_directory) == "fallout4.exe not found":
                w = MessageBox(title="Fallout 4 Not Found", content="Fallout 4 was not found installed on your computer. Please go to settings and set the Fallout 4 install directory to get the full features of this app.", parent=self)
                w.cancelButton.setText(self.tr("Do Not Remind Me Again"))
                if w.exec():
                    pass
                else:
                    cfg.set(cfg.fallout_4_directory_check, False)
            else:
                cfg.set(cfg.fallout_4_directory_check, False)

    def verify(self):
        title = 'Disclaimer for Use of for FallTalk'
        content = DISCLAIMER
        w = MessageBox(title, content, self)
        w.contentLabel.setMinimumHeight(675)
        w.contentLabel.setMinimumWidth(1100)
        w.contentLabel.setText(w.content)
        w.contentLabel.setWordWrap(True)
        w.widget.setFixedSize(
            max(w.contentLabel.width(), w.titleLabel.width()) + 48,
            w.contentLabel.y() + w.contentLabel.height() + 105
        )
        w.yesButton.setText(self.tr('Agree'))
        if w.exec():
            cfg.set(cfg.accepted_disclaimer, True)
        else:
            sys.exit()

    def downloadModels(self):
        title = 'Download Engines'
        content = """Would you like to download all the engines? This will be about 10 GBs and take a few minutes. They will be downloaded on demand as needed otherwise."""
        w = MessageBox(title, content, self)
        w.yesButton.setText(self.tr('Download'))
        w.cancelButton.setText(self.tr('No'))
        if w.exec():
            self.showLoaderPopup("Downloading", "Please Wait")
            tr = (threading.Thread(target=falltalkutils.downloadBaseModels, args={self}, daemon=True))
            tr.start()

    def download_models_config(self):
        try:
            os.makedirs("models/", exist_ok=True)
            os.makedirs("config/", exist_ok=True)
            old_json = None
            if os.path.exists('config/models.json'):
                with open('config/models.json', 'r', encoding="utf-8") as old_file:
                    old_json = json.load(old_file)
            huggingface_hub.hf_hub_download(REPO, "config/models.json", local_dir=os.path.abspath(f"./"))

            with open('config/models.json', 'r', encoding="utf-8") as new_file:
                new_json = json.load(new_file)

            diff = falltalkutils.get_model_diff(old_json, new_json)

            if diff is not None:
                self.toolbar_1.addSeparator()
                self.toolbar_1.addAction(self.new_models_action)
                w = MessageBox("Updates", diff, self)
                self.new_models_action.triggered.connect(lambda: w.open())

            huggingface_hub.hf_hub_download(REPO, "config/characters.json", local_dir=os.path.abspath(f"./"))
        except Exception as e:
            falltalkutils.logger.exception("Unable to load configs")
            if not os.path.exists('config/characters.json') and not os.path.exists('config/models.json'):
                self.createErrorInfoBar("Unable to Download Configs", "Unable to download the required configuration files, please check your network")

    def xttsv2_checked(self, checked):
        if checked:
            cfg.set(cfg.engine, "XTTSv2")
            self.onEngineChange(cfg.engine)
        elif cfg.get(cfg.engine) == "XTTSv2":
            self.xtts_action.setChecked(True)

    def voicecraft_checked(self, checked):
        if checked:
            cfg.set(cfg.engine, "VoiceCraft")
            self.onEngineChange(cfg.engine)
        elif cfg.get(cfg.engine) == "VoiceCraft":
            self.voicecraft_action.setChecked(True)

    def rvc_checked(self, checked):
        if checked:
            cfg.set(cfg.engine, "RVC")
            self.onEngineChange(cfg.engine)
        elif cfg.get(cfg.engine) == "RVC":
            self.rvc_action.setChecked(True)

    def styletts2_checked(self, checked):
        if checked:
            cfg.set(cfg.engine, "StyleTTS2")
            self.onEngineChange(cfg.engine)
        elif cfg.get(cfg.engine) == "StyleTTS2":
            self.styletts2_action.setChecked(True)

    def gpt_sovits_checked(self, checked):
        if checked:
            cfg.set(cfg.engine, "GPT_SoVITS")
            self.onEngineChange(cfg.engine)
        elif cfg.get(cfg.engine) == "GPT_SoVITS":
            self.gpt_sovits_action.setChecked(True)

    def cpu_checked(self, checked):
        if checked:
            cfg.set(cfg.device, "cpu")
            self.gpu_action.setChecked(False)
            self.gpu2_action.setChecked(False)
            self.onDeviceChange()
        elif cfg.get(cfg.device) == "cpu":
            self.cpu_action.setChecked(True)

    def gpu_checked(self, checked):
        if checked:
            cfg.set(cfg.device, "cuda")
            self.cpu_action.setChecked(False)
            self.gpu2_action.setChecked(False)
            self.onDeviceChange()
        elif cfg.get(cfg.device) == "cuda":
            self.gpu_action.setChecked(True)

    def gpu2_checked(self, checked):
        if checked:
            cfg.set(cfg.device, "cuda:1")
            self.gpu_action.setChecked(False)
            self.cpu_action.setChecked(False)
            self.onDeviceChange()
        elif cfg.get(cfg.device) == "cuda:1":
            self.gpu2_action.setChecked(True)

    def initUI(self):

        self.xtts_widget = XttsWidget(self)
        self.voicecraft_widget = VoiceCraftWidget(self)
        self.faq_widget = FaqWidget(self)
        self.gpt_sovits_widget = GPT_SoVITSWidget(self)
        self.styletts2_widget = StyleTTS2Widget(self)
        self.rvc_widget = RVCWidget(self)
        self.rvc_widget.rvc_mic_widget.media_recorder.doneRecording.connect(self.generate_audio)

        self.characters_widget = CharactersWidget(self)
        self.reference_widget = ReferencesWidget(self)
        self.generate_widget = FallTalkWidget(parent=self, text="Generate Voice")

        self.generate_widget.addToFrame(self.xtts_widget)
        self.generate_widget.addToFrame(self.voicecraft_widget)
        self.generate_widget.addToFrame(self.gpt_sovits_widget)
        self.generate_widget.addToFrame(self.styletts2_widget)
        self.generate_widget.addToFrame(self.rvc_widget)

        self.bulk_generate_widget = BulkGenerationWidget(parent=self)

        self.upscale_widget = UpscaleWidget(parent=self)
        self.music_widget = MusicGenWidget(parent=self)
        self.fx_widget = AudioGenWidget(parent=self)

        self.setting_widget = SettingsWidget(self)

        self.addSubInterface(self.characters_widget, FIF.PEOPLE, 'Character Models', NavigationItemPosition.TOP)
        self.addSubInterface(self.reference_widget, FIF.MIX_VOLUMES, 'Reference Audio', NavigationItemPosition.TOP)
        self.addSubInterface(self.generate_widget, FIF.ROBOT, 'Generate Voice', NavigationItemPosition.TOP)
        self.navigationInterface.addSeparator()
        self.addSubInterface(self.bulk_generate_widget, FallTalkIcons.BULK.icon(), 'Bulk Generation', NavigationItemPosition.TOP)
        self.addSubInterface(self.upscale_widget, FallTalkIcons.ENHANCE.icon(), 'Bulk Enhancement', NavigationItemPosition.TOP)
        self.navigationInterface.addSeparator()
        self.addSubInterface(self.music_widget, FallTalkStrokeIcons.MUSIC.icon(), 'Music Generator', NavigationItemPosition.SCROLL)
        self.addSubInterface(self.fx_widget, FallTalkIcons.FX.icon(), 'Sound FX Generator', NavigationItemPosition.SCROLL)

        self.addSubInterface(self.faq_widget, FIF.HELP, 'FAQ', NavigationItemPosition.BOTTOM)
        self.addSubInterface(self.setting_widget, FIF.SETTING, 'Settings', NavigationItemPosition.BOTTOM)

        self.rvc_action.triggered.connect(self.rvc_checked)
        self.gpt_sovits_action.triggered.connect(self.gpt_sovits_checked)
        self.voicecraft_action.triggered.connect(self.voicecraft_checked)
        self.xtts_action.triggered.connect(self.xttsv2_checked)
        self.styletts2_action.triggered.connect(self.styletts2_checked)

        self.cpu_action.triggered.connect(self.cpu_checked)
        self.gpu_action.triggered.connect(self.gpu_checked)
        self.gpu2_action.triggered.connect(self.gpu2_checked)
        self.clean_action.triggered.connect(self.clean_engines)

        QApplication.processEvents()

    @Slot(PySide6.QtCore.QObject, str, str)
    def onError(self, parent, title, text):
        parent.complete_loader()
        parent.createErrorInfoBar(title, text)

    @Slot(PySide6.QtCore.QObject, str, str)
    def onWarn(self, parent, title, text):
        parent.createErrorInfoBar(title, text)

    def createErrorInfoBar(self, title, text):
        InfoBar.error(
            title=title,
            content=text,
            orient=Qt.Orientation.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=10000,
            parent=self
        )

    def showLoaderPopup(self, title, text):
        self.setEnabled(False)
        self.stateTooltip = StateToolTip(title, text, self)
        self.stateTooltip.setMinimumWidth(350)
        self.stateTooltip.closeButton.setVisible(False)
        label_size = self.stateTooltip.width()
        window_width = self.width()
        x = (window_width - label_size) // 2
        self.stateTooltip.move(x, 10)
        self.stateTooltip.show()

    @Slot(str)
    def update_loader(self, content):
        self.stateTooltip.setContent(content)

    def complete_loader(self):
        self.stateTooltip.setContent("Completed")
        self.stateTooltip.setState(True)
        self.stateTooltip = None
        self.setEnabled(True)

    def clean_engines(self):
        if self.sound_fx_engine is not None:
            self.sound_fx_engine.clean()
            del self.sound_fx_engine
            self.sound_fx_engine = None

        if self.music_engine is not None:
            self.music_engine.clean()
            del self.music_engine
            self.music_engine = None

        print("Cache Cleaned")

    def onDeviceChange(self):
        falltalkutils.logger.debug(f'Device Changed {cfg.get(cfg.device)}')
        tr = (threading.Thread(target=self.clean_engines, daemon=True))
        tr.start()

        if self.tts_engine is not None:
            self.onEngineChange(cfg.get(cfg.engine))

    @Slot(str)
    def engine_change(self, engine):
        cfg.set(cfg.engine, engine)
        self.onEngineChange(cfg.get(cfg.engine))

    def onEngineChange(self, engine):
        self.showLoaderPopup("Loading Engine", f"Loading {engine.value}")
        falltalkutils.logger.debug(f"Engine Changed to {engine.value}")

        self.voicecraft_widget.setVisible(False)
        self.voicecraft_widget.setEnabled(False)
        self.gpt_sovits_widget.setEnabled(False)
        self.gpt_sovits_widget.setVisible(False)
        self.xtts_widget.setVisible(False)
        self.xtts_widget.setEnabled(False)
        self.styletts2_widget.setVisible(False)
        self.styletts2_widget.setEnabled(False)
        self.rvc_widget.setVisible(False)
        self.rvc_widget.setEnabled(False)
        self.bulk_generate_widget.setEnabled(False)

        self.character_label.setText(f"Please Load Model")

        self.rvc_action.setChecked(False)
        self.gpt_sovits_action.setChecked(False)
        self.voicecraft_action.setChecked(False)
        self.xtts_action.setChecked(False)
        self.styletts2_action.setChecked(False)

        if self.tts_engine is not None:
            self.tts_engine.clean()
        if engine.value == "XTTSv2":
            self.xtts_action.setChecked(True)
            tr = (threading.Thread(target=falltalkutils.load_xtts, args={self}, daemon=True))
            tr.start()
        elif engine.value == "VoiceCraft":
            self.voicecraft_action.setChecked(True)
            tr = (threading.Thread(target=falltalkutils.load_voicecraft, args={self}, daemon=True))
            tr.start()
        elif engine.value == "GPT_SoVITS":
            self.gpt_sovits_action.setChecked(True)
            tr = (threading.Thread(target=falltalkutils.load_gpt_sovits, args={self}, daemon=True))
            tr.start()
        elif engine.value == "StyleTTS2":
            self.styletts2_action.setChecked(True)
            tr = (threading.Thread(target=falltalkutils.load_style_tts2, args={self}, daemon=True))
            tr.start()
        elif engine.value == "RVC":
            self.rvc_action.setChecked(True)
            tr = (threading.Thread(target=falltalkutils.load_rvc, args={self}, daemon=True))
            tr.start()
        else:
            self.tts_engine = None
            self.complete_loader()

    @Slot(PySide6.QtCore.QObject, str)
    def updateMediaplayer(self, panel, url):
        panel.media_player.player.stop()
        QTimer.singleShot(0, lambda: (
            panel.media_player.player.setSource(QUrl.fromLocalFile(url)),
            cfg.get(cfg.auto_play) and panel.media_player.player.play()
        ))

    @Slot(PySide6.QtCore.QObject)
    def afterXtts(self, parent):
        parent.complete_loader()
        parent.xtts_widget.setVisible(True)
        parent.xtts_widget.setEnabled(True)
        parent.xtts_widget.media_player.setVisible(True)
        parent.bulk_generate_widget.setEnabled(True)

    @Slot(PySide6.QtCore.QObject)
    def afterVoiceCraft(self, parent):
        parent.complete_loader()
        parent.voicecraft_widget.setEnabled(True)
        parent.voicecraft_widget.setVisible(True)
        parent.voicecraft_widget.media_player.setVisible(True)
        parent.bulk_generate_widget.setEnabled(False)

    @Slot(PySide6.QtCore.QObject)
    def afterGPT_SoVITS(self, parent):
        parent.complete_loader()
        parent.gpt_sovits_widget.setEnabled(True)
        parent.gpt_sovits_widget.setVisible(True)
        parent.gpt_sovits_widget.media_player.setVisible(True)
        parent.bulk_generate_widget.setEnabled(True)

    @Slot(PySide6.QtCore.QObject)
    def afterStyleTTS2(self, parent):
        parent.complete_loader()
        parent.styletts2_widget.setEnabled(True)
        parent.styletts2_widget.setVisible(True)
        parent.styletts2_widget.media_player.setVisible(True)
        parent.bulk_generate_widget.setEnabled(True)

    @Slot(PySide6.QtCore.QObject)
    def afterRVC(self, parent):
        parent.complete_loader()
        parent.rvc_widget.setEnabled(True)
        parent.rvc_widget.setVisible(True)
        parent.rvc_widget.media_player.setVisible(True)
        parent.bulk_generate_widget.setEnabled(True)

    @Slot(PySide6.QtCore.QObject)
    def afterModelLoader(self, parent):
        parent.complete_loader()
        if cfg.get(cfg.engine) == "RVC":
            self.stackedWidget.setCurrentWidget(self.generate_widget)
        elif cfg.get(cfg.engine) == "GPT_SoVITS":
            if parent.tts_engine.is_base:
                parent.gpt_sovits_widget.text_input.setPlaceholderText("Please Select the 'Transcribe Reference Audio' button below")
                parent.gpt_sovits_widget.transcribe_button.setVisible(True)
            else:
                parent.gpt_sovits_widget.text_input.setPlaceholderText("Please Enter Text")
                parent.gpt_sovits_widget.transcribe_button.setVisible(False)

            self.stackedWidget.setCurrentWidget(self.reference_widget)
        else:
            self.stackedWidget.setCurrentWidget(self.reference_widget)

    @Slot(PySide6.QtCore.QObject)
    def afterGen(self, parent):
        parent.complete_loader()

    @Slot(PySide6.QtCore.QObject)
    def afterDownload(self, parent):
        parent.complete_loader()

    @Slot(PySide6.QtCore.QObject)
    def afterModelDownload(self, parent):
        parent.complete_loader()
        parent.load_models_config()

    @Slot(PySide6.QtCore.QObject)
    def afterModelConfigLoad(self, parent):
        parent.complete_loader()

    @Slot(PySide6.QtCore.QObject)
    def continueLoad(self, parent):
        self.bulk_generate_widget.setEnabled(True)
        parent.load_models_config()
        if parent.pending_bulk:
            parent.bulk_inference()
        elif parent.pending_character and not parent.pending_base:
            parent.stackedWidget.setCurrentWidget(parent.characters_widget)
            parent.load_trained_model(parent.pending_character, parent.pending_model, parent.pending_rvc)
        else:
            parent.stackedWidget.setCurrentWidget(parent.characters_widget)
            parent.load_base_model(parent.pending_character, parent.pending_rvc)

    def find_first_match_by_name(self, models, name):
        for character in models['characters']:
            if character['name'] == name:
                return character
        return None

    def load_models_config(self):
        if self.characters_data is None:
            with open('config/characters.json', 'r', encoding='utf-8') as file:
                self.characters_data = {character['name']: character for character in json.load(file)}
        if self.models is None:
            with open('config/models.json', 'r', encoding='utf-8') as file:
                self.models = {model['name']: model for model in json.load(file)['characters']}

        if os.path.exists('config/custom_models.json'):
            with open('config/custom_models.json', 'r', encoding="utf-8") as file:
                self.custom_models = {model['name']: model for model in json.load(file)}

        trained_characters = []
        untrained_characters = []

        for character_name, character in self.characters_data.items():
            model_found = False
            character['RVC'] = None
            if character_name in self.models:
                character['display_name'] = self.models[character_name]['display_name']
                if "RVC" in self.models[character_name]:
                    character['RVC'] = self.models[character_name]['RVC']

                if cfg.get(cfg.engine) in self.models[character_name]:
                    model_found = True
                    c = copy.copy(character)
                    c['display_name'] = self.models[character_name]['display_name']
                    c[cfg.get(cfg.engine)] = self.models[character_name][cfg.get(cfg.engine)]
                    trained_characters.append(c)

            if not model_found:
                if 'display_name' not in character:
                    character['display_name'] = character['name']
                untrained_characters.append(character)

        self.characters_widget.clear()
        self.characters_widget.loadTrained(self, trained_characters)
        if cfg.get(cfg.engine) != 'RVC':
            self.characters_widget.loadUntrained(self, untrained_characters)

        if self.custom_models is not None:
            self.characters_widget.loadCustom(self, self.custom_models)

        self.bulk_generate_widget.populate_character_card()

    def update_reference_table(self, selected_model):
        if selected_model:
            self.reference_widget.addDataToReferencesTable(selected_model)

    def transcribe(self, widget):
        references = self.reference_widget.reference_audio
        if references is None or not references:
            self.showErrorPopup(widget, widget.transcribe_button, "Please Select Reference Audio")
        else:
            if len(references) == 1:
                falltalkutils.logger.debug(references)
                selected_audio = references[0]
            else:
                falltalkutils.logger.debug(references)
                selected_audio = falltalkutils.combine_wav_files(references)

            self.showLoaderPopup("Transcribing Audio", "Please Wait")
            tr = (threading.Thread(target=falltalkutils.do_transcribe, args=(self, selected_audio, widget), daemon=True))
            tr.start()

    @Slot(PySide6.QtCore.QObject, PySide6.QtCore.QObject)
    def after_transcribe(self, parent, widget):
        widget.clear()
        widget.load_data()
        parent.complete_loader()

    def combine_references(self, references):
        falltalkutils.logger.debug(f"{references}")
        if len(references) == 1:
            falltalkutils.logger.debug(references)
            selected_audio = references[0]
        else:
            falltalkutils.logger.debug(references)
            selected_audio = falltalkutils.combine_wav_files(references)

        return selected_audio

    def get_output_file_name(self, file_name):
        if file_name is None or file_name == "" or file_name == "Random":
            unique_id = uuid.uuid4()
            file_name = f"{falltalkutils.formatted_time_stamp()}_{self.tts_engine.model_name}_{self.tts_engine.engine_name}_{unique_id.hex[:10]}"

        path = os.path.abspath(os.path.join(cfg.get(cfg.output_dir), self.tts_engine.model_name, f"{file_name}.wav"))
        os.makedirs(os.path.join(cfg.get(cfg.output_dir), self.tts_engine.model_name), exist_ok=True)
        return path

    def get_music_file(self, file_name, wav=False):
        if file_name is None or file_name == "" or file_name == "Random":
            file_name = falltalkutils.formatted_time_stamp_uuid()

        path = os.path.abspath(os.path.join(cfg.get(cfg.output_dir), "music", f"{file_name}.wav" if wav else f"{file_name}"))
        os.makedirs(os.path.join(cfg.get(cfg.output_dir), "music"), exist_ok=True)
        return path

    def get_fx_file(self, file_name, wav=False):
        if file_name is None or file_name == "" or file_name == "Random":
            file_name = falltalkutils.formatted_time_stamp_uuid()

        path = os.path.abspath(os.path.join(cfg.get(cfg.output_dir), "fx", f"{file_name}.wav" if wav else f"{file_name}"))
        os.makedirs(os.path.join(cfg.get(cfg.output_dir), "fx"), exist_ok=True)
        return path

    def get_output_file(self, widget):
        return self.get_output_file_name(widget.output_name.value)

    def ensure_sentence_punctuation(self, sentence):
        if not sentence:
            return None
        # Define the standard punctuation marks
        standard_punctuation = ".!?"

        sentence = sentence.strip()

        # Check if the last character of the sentence is a standard punctuation mark
        if sentence[-1] not in standard_punctuation:
            # If not, add a period at the end
            sentence += "."

        return sentence

    @Slot(PySide6.QtCore.QObject)
    def after_fx_gen(self, parent):
        parent.complete_loader()
        parent.generate_fx()

    def generate_fx(self):
        if self.sound_fx_engine is None:
            self.showLoaderPopup("Loading FX Generator", "Please Wait")
            tr = (threading.Thread(target=falltalkutils.load_fx_gen, args={self}, daemon=True))
            tr.start()
        else:
            output_file = str(self.get_fx_file(self.fx_widget.output_name.value, True))
            text = self.fx_widget.text_input.toPlainText()
            if text is None or text == "":
                self.showErrorPopup(self.fx_widget, self.fx_widget.generate_button, "Please Enter Valid Text")
            else:
                self.showLoaderPopup("Generating Sound FX", "Please Wait")
                tr = (threading.Thread(target=self.sound_fx_engine.generate, args=(text, output_file, self.fx_widget, cfg.fx_duration.value), daemon=True))
                tr.start()

    @Slot(PySide6.QtCore.QObject)
    def after_music_gen(self, parent):
        parent.complete_loader()
        parent.generate_music()

    def generate_music(self):
        if self.music_engine is None:
            self.showLoaderPopup("Loading Music Generator", "Please Wait")
            tr = (threading.Thread(target=falltalkutils.load_music_gen, args={self}, daemon=True))
            tr.start()
        else:
            output_file = str(self.get_music_file(self.music_widget.output_name.value))
            text = self.music_widget.text_input.toPlainText()
            if text is None or text == "":
                self.showErrorPopup(self.music_widget, self.music_widget.generate_button, "Please Enter Valid Text")
            else:
                ref = self.music_widget.ref_file.value if self.music_widget.ref_file.value != "Please Select a File" else None

                self.showLoaderPopup("Generating Music", "Please Wait")
                tr = (threading.Thread(target=self.music_engine.generate, args=(text, output_file, self.music_widget, cfg.audio_mode.value, cfg.music_duration.value, ref), daemon=True))
                tr.start()

    @Slot(PySide6.QtCore.QObject)
    def after_upscale(self, parent):
        parent.complete_loader()
        parent.upscale_folder()

    def upscale_folder(self):
        if self.upscale_engine is None:
            self.showLoaderPopup("Loading Enhancer", "Please Wait")
            tr = (threading.Thread(target=falltalkutils.load_upscaler, args={self}, daemon=True))
            tr.start()
        else:
            up_dir = self.upscale_widget.upscale_dir.value
            if up_dir is None or up_dir == "Please Select a Valid Folder":
                self.showErrorPopup(self.upscale_widget, self.upscale_widget.generate_button, "Please Select a Valid Folder")
            else:
                self.showLoaderPopup("Enhancing", "Please Wait")
                tr = (threading.Thread(target=self.upscale_engine.upscale_dir, args=(up_dir, cfg.get(cfg.replace_existing), cfg.get(cfg.include_subdir), self.upscale_widget.audio_mode.value, self.upscale_widget.sample_rate.value), daemon=True))
                tr.start()



    def bulk_inference(self):
        # if self.bulk_generate_widget.stackedWidget.currentWidget() == self.bulk_generate_widget.bulk_rvc_widget and (self.tts_engine is None or self.tts_engine.engine_name != 'RVC'):
        #     self.pending_bulk = True
        #     cfg.set(cfg.engine, 'RVC')
        #     self.onEngineChange(cfg.engine)
        if self.bulk_generate_widget.stackedWidget.currentWidget() == self.bulk_generate_widget.bulk_csv_widget and self.tts_engine is None:
            self.pending_bulk = True
            self.onEngineChange(cfg.engine)
        else:
            self.pending_bulk = False
            if self.bulk_generate_widget.stackedWidget.currentWidget() == self.bulk_generate_widget.bulk_csv_widget:
                data = len(self.bulk_generate_widget.bulk_csv_widget.bulk_table.model().getData())
                if data > 0:
                    self.showLoaderPopup(f"Generating Bulk Audio", f"Completed: 0/{data}")
                    tr = (threading.Thread(target=falltalkutils.bulk_inference, args={self}, daemon=True))
                    tr.start()
                else:
                    self.showErrorPopup(self.bulk_generate_widget, self.bulk_generate_widget.generate_button, "Please Select Load some Data")
            elif self.bulk_generate_widget.stackedWidget.currentWidget() == self.bulk_generate_widget.bulk_rvc_widget:
                rvc_dir = self.bulk_generate_widget.bulk_rvc_widget.rvc_dir.value
                if rvc_dir is None or rvc_dir == "Please Select a Valid Folder":
                    self.showErrorPopup(self.bulk_generate_widget, self.bulk_generate_widget.generate_button, "Please Select a Valid Folder")
                elif self.bulk_generate_widget.bulk_rvc_widget.character_card.configItem.text() is None or self.bulk_generate_widget.bulk_rvc_widget.character_card.configItem.text() == '':
                    self.showErrorPopup(self.bulk_generate_widget, self.bulk_generate_widget.generate_button, "Please Select a valid Character")
                else:
                    self.showLoaderPopup(f"Generating Bulk RVC Audio", f"Gathering Files")
                    tr = (threading.Thread(target=falltalkutils.bulk_rvc_inference, args=(self, rvc_dir, self.bulk_generate_widget.bulk_rvc_widget.character_card.configItem.text(), cfg.get(cfg.include_subdir), cfg.get(cfg.replace_existing), cfg.get(cfg.threads), cfg.get(cfg.use_existing_lip)), daemon=True))
                    tr.start()
            elif self.bulk_generate_widget.stackedWidget.currentWidget() == self.bulk_generate_widget.bulk_fuz_widget:
                lip_dir = self.bulk_generate_widget.bulk_fuz_widget.lip_dir.value
                if lip_dir is None or lip_dir == "Please Select a Valid Folder":
                    self.showErrorPopup(self.bulk_generate_widget, self.bulk_generate_widget.generate_button, "Please Select a Valid Folder")
                else:
                    self.showLoaderPopup(f"Generating Bulk FUZ Audio", f"Gathering Files")
                    tr = (threading.Thread(target=falltalkutils.bulk_fuz, args=(self, lip_dir, cfg.get(cfg.include_subdir), cfg.get(cfg.replace_existing), cfg.get(cfg.threads)), daemon=True))
                    tr.start()



    def generate_audio(self, recording_file=None):
        references = self.reference_widget.reference_audio
        references_length = self.reference_widget.reference_audio_length
        if cfg.get(cfg.engine) == "RVC":
            if self.rvc_widget.stackedWidget.currentWidget() == self.rvc_widget.rvc_mic_widget:
                if recording_file is None:
                    self.showErrorPopup(self.rvc_widget, self.rvc_widget.rvc_mic_widget.media_recorder.recordButton, "Please Select Record Audio")
                elif self.tts_engine.model_name is None:
                    self.showErrorPopup(self.rvc_widget, self.rvc_widget.rvc_mic_widget.recordButton, "Please Select Load a Model")
                else:
                    self.showLoaderPopup("Cloning Input Audio", "Please Wait")
                    output_file = self.get_output_file(self.rvc_widget.rvc_mic_widget)
                    shutil.copy(recording_file, output_file)
                    tr = (threading.Thread(target=falltalkutils.rvc_inference, args=(self, output_file, self.rvc_widget), daemon=True))
                    tr.start()
            elif self.rvc_widget.stackedWidget.currentWidget() == self.rvc_widget.rvc_file_widget:
                recording_file = self.rvc_widget.rvc_file_widget.rvc_file.value
                if recording_file is None:
                    self.showErrorPopup(self.rvc_widget, self.rvc_widget.rvc_file_widget, "Please Select a file")
                elif self.tts_engine.model_name is None:
                    self.showErrorPopup(self.rvc_widget, self.rvc_widget.rvc_file_widget, "Please Select Load a Model")
                else:
                    self.showLoaderPopup("Cloning File Audio", "Please Wait")
                    output_file = self.get_output_file(self.rvc_widget.rvc_file_widget)
                    shutil.copy(recording_file, output_file)
                    tr = (threading.Thread(target=falltalkutils.rvc_inference, args=(self, output_file, self.rvc_widget), daemon=True))
                    tr.start()
            elif self.rvc_widget.stackedWidget.currentWidget() == self.rvc_widget.edge_tts_widget:
                text = falltalkutils.replace_numbers_with_words(self.ensure_sentence_punctuation(self.rvc_widget.edge_tts_widget.text_input.toPlainText()))
                if not text or text == '':
                    self.showErrorPopup(self.rvc_widget, self.rvc_widget.edge_tts_widget.generate_button, "Please Enter Some Text to Generate...")
                elif self.tts_engine.model_name is None:
                    self.showErrorPopup(self.rvc_widget, self.rvc_widget.edge_tts_widget.generate_button, "Please Select Load a Model")
                else:
                    self.showLoaderPopup("Cloning TTS from Edge", "Please Wait")
                    output_file = self.get_output_file(self.rvc_widget.edge_tts_widget)
                    tr = (threading.Thread(target=falltalkutils.edge_tts_inference, args=(self, text, output_file, self.rvc_widget.edge_tts_widget.voice_combo.configItem.currentText(), self.rvc_widget), daemon=True))
                    tr.start()
            elif self.rvc_widget.stackedWidget.currentWidget() == self.rvc_widget.eleven_labs_widget:
                text = falltalkutils.replace_numbers_with_words(self.ensure_sentence_punctuation(self.rvc_widget.eleven_labs_widget.text_input.toPlainText()))
                if not text or text == '':
                    self.showErrorPopup(self.rvc_widget, self.rvc_widget.eleven_labs_widget.generate_button, "Please Enter Some Text to Generate...")
                elif self.tts_engine.model_name is None:
                    self.showErrorPopup(self.rvc_widget, self.rvc_widget.eleven_labs_widget.generate_button, "Please Select Load a Model")
                else:
                    self.showLoaderPopup("Calling ElevenLabs", "Please Wait")
                    output_file = self.get_output_file(self.rvc_widget.eleven_labs_widget)
                    tr = (threading.Thread(target=falltalkutils.eleven_labs_inference, args=(self, text, output_file, self.rvc_widget.eleven_labs_widget.voice_combo.configItem.currentText(), self.rvc_widget), daemon=True))
                    tr.start()
        elif cfg.get(cfg.engine) == "XTTSv2":
            text = falltalkutils.replace_numbers_with_words(self.ensure_sentence_punctuation(self.xtts_widget.text_input.toPlainText()))
            if references is None or not references:
                self.showErrorPopup(self.xtts_widget, self.xtts_widget.generate_button, "Please Select Reference Audio")
            elif references_length < 10:
                self.showErrorPopup(self.xtts_widget, self.xtts_widget.generate_button, "Please Select at least 10 seconds of Reference Audio")
            elif not text or text == '':
                self.showErrorPopup(self.xtts_widget, self.xtts_widget.generate_button, "Please Enter Some Text to Generate...")
            else:
                self.showLoaderPopup("Generating Audio", "Please Wait")
                tr = (threading.Thread(target=falltalkutils.xtts_inference, args=(self, self.get_output_file(self.xtts_widget), text, self.combine_references(references), self.xtts_widget), daemon=True))
                tr.start()
        elif cfg.get(cfg.engine) == "VoiceCraft":
            text = falltalkutils.replace_numbers_with_words(self.ensure_sentence_punctuation(self.voicecraft_widget.text_input.toPlainText()))
            start_word = self.voicecraft_widget.start_dropdown_card.getWordInfo()
            end_word = self.voicecraft_widget.end_dropdown_card.getWordInfo()
            start_tts_word = self.voicecraft_widget.start_tts_dropdown_card.getWordInfo()
            start_time = start_word['start'] if start_word is not None else None
            end_time = end_word['end'] if end_word is not None else None
            start_tts_time = start_tts_word['end'] if start_tts_word is not None else None
            if references is None or not references:
                self.showErrorPopup(self.voicecraft_widget, self.voicecraft_widget.generate_button, "Please Select Reference Audio")
            elif references_length > 16:
                self.showErrorPopup(self.xtts_widget, self.xtts_widget.generate_button, "Please Select less than 16 seconds of Reference Audio")
            elif start_time is None and end_time is None and start_tts_time is None:
                self.showErrorPopup(self.voicecraft_widget, self.voicecraft_widget.generate_button, "Please Transcribe your reference audio")
            elif not text or text == '':
                self.showErrorPopup(self.voicecraft_widget, self.voicecraft_widget.generate_button, "Please Enter Some Text to Generate...")
            elif cfg.get(cfg.mode) == "edit" and start_time > end_time:
                self.showErrorPopup(self.voicecraft_widget, self.voicecraft_widget.generate_button, "Start must come before the end time")
            else:
                self.showLoaderPopup("Generating Audio", "Please Wait")
                tr = (threading.Thread(target=falltalkutils.voicecraft_inference, args=(self, self.get_output_file(self.voicecraft_widget), text, self.combine_references(references), self.voicecraft_widget, start_time, end_time, start_tts_time, self.voicecraft_widget.transcribe_state), daemon=True))
                tr.start()
        elif cfg.get(cfg.engine) == "GPT_SoVITS":
            text = falltalkutils.replace_numbers_with_words(self.ensure_sentence_punctuation(self.gpt_sovits_widget.text_input.toPlainText()))
            transcribe_state = None
            if self.tts_engine.is_base:
                transcribe_state = self.gpt_sovits_widget.transcribe_state
            if references is None or not references:
                self.showErrorPopup(self.gpt_sovits_widget, self.gpt_sovits_widget.generate_button, "Please Select Reference Audio")
            elif self.tts_engine.is_base and (references_length > 15 or references_length < 3):
                self.showErrorPopup(self.xtts_widget, self.xtts_widget.generate_button, "Please Select between 3 and 10 seconds of Reference Audio")
            elif not self.tts_engine.is_base and references_length < 3:
                self.showErrorPopup(self.xtts_widget, self.xtts_widget.generate_button, "Please Select at least 3 seconds of Reference Audio")
            elif self.tts_engine.is_base and transcribe_state is None:
                self.showErrorPopup(self.gpt_sovits_widget, self.gpt_sovits_widget.generate_button, "Please Transcribe your reference audio")
            elif not text or text == '':
                self.showErrorPopup(self.gpt_sovits_widget, self.gpt_sovits_widget.generate_button, "Please Enter Some Text to Generate...")
            else:
                self.showLoaderPopup("Generating Audio", "Please Wait")
                tr = (threading.Thread(target=falltalkutils.gpt_sovits_inference, args=(self, self.get_output_file(self.gpt_sovits_widget), text, self.combine_references(references) if self.tts_engine.is_base else references, self.gpt_sovits_widget, transcribe_state), daemon=True))
                tr.start()
        elif cfg.get(cfg.engine) == "StyleTTS2":
            text = falltalkutils.replace_numbers_with_words(self.ensure_sentence_punctuation(self.styletts2_widget.text_input.toPlainText()))
            if references_length > 15 or references_length < 5:
                self.showErrorPopup(self.xtts_widget, self.xtts_widget.generate_button, "Please Select between 5 and 10 seconds of Reference Audio")
            elif references is None or not references:
                self.showErrorPopup(self.styletts2_widget, self.styletts2_widget.generate_button, "Please Select Reference Audio")
            elif not text or text == '':
                self.showErrorPopup(self.styletts2_widget, self.styletts2_widget.generate_button, "Please Enter Some Text to Generate...")
            else:
                self.showLoaderPopup("Generating Audio", "Please Wait")
                tr = (threading.Thread(target=falltalkutils.styletts2_inference, args=(self, self.get_output_file(self.styletts2_widget), text, self.combine_references(references), self.styletts2_widget), daemon=True))
                tr.start()

    def showErrorPopup(self, parent, target, content):
        Flyout.create(
            icon=InfoBarIcon.ERROR,
            title='Error',
            content=content,
            target=target,
            parent=parent,
            isClosable=True
        )

    def load_base_model(self, character, rvc, base_model=True):
        if character:
            if cfg.get(cfg.fallout_4_directory) == "fallout4.exe not found":
                self.verifyFallout()
            else:
                self.reference_widget.addDataToReferencesTable(character)

        if self.tts_engine is None and cfg.engine is not None:
            self.pending_character = character
            self.pending_rvc = rvc
            self.pending_base = True
            self.onEngineChange(cfg.engine)
        elif character:
            if cfg.get(cfg.engine) == "XTTSv2":
                self.xtts_widget.rvc_enabled.setVisible(rvc is not None)
            elif cfg.get(cfg.engine) == "VoiceCraft":
                self.voicecraft_widget.rvc_enabled.setVisible(rvc is not None)
            elif cfg.get(cfg.engine) == "GPT_SoVITS":
                self.gpt_sovits_widget.rvc_enabled.setVisible(rvc is not None)
            elif cfg.get(cfg.engine) == "StyleTTS2":
                self.styletts2_widget.rvc_enabled.setVisible(rvc is not None)

            self.showLoaderPopup("Loading Base Model", f"Loading")
            tr = (threading.Thread(target=falltalkutils.load_model, args=(self, character['name'], rvc, character['display_name'], base_model), daemon=True))
            tr.start()

    def load_custom_model(self, character, model, rvc):
        if cfg.get(cfg.engine) in model:
            self.load_trained_model(character, model, rvc)
        else:
            self.load_base_model(character, model, rvc)

    def load_trained_model(self, character, model, rvc):
        if self.tts_engine is None and cfg.engine is not None:
            self.pending_character = character
            self.pending_model = model
            self.pending_rvc = rvc
            self.pending_base = False
            self.onEngineChange(cfg.engine)
        else:
            if model:
                if cfg.get(cfg.fallout_4_directory) == "fallout4.exe not found":
                    self.verifyFallout()
                else:
                    self.reference_widget.addDataToReferencesTable(model)
            self.update_reference_table(model)
            self.pending_character = None
            self.pending_model = None
            self.pending_rvc = None

            if cfg.get(cfg.engine) == "XTTSv2":
                self.xtts_widget.rvc_enabled.setVisible(rvc is not None)
            elif cfg.get(cfg.engine) == "VoiceCraft":
                self.voicecraft_widget.rvc_enabled.setVisible(rvc is not None)
            elif cfg.get(cfg.engine) == "GPT_SoVITS":
                self.gpt_sovits_widget.rvc_enabled.setVisible(rvc is not None)
            elif cfg.get(cfg.engine) == "StyleTTS2":
                self.styletts2_widget.rvc_enabled.setVisible(rvc is not None)

            self.showLoaderPopup("Loading Model", f"Loading {character}")
            tr = (threading.Thread(target=falltalkutils.load_model, args=(self, character, rvc, model['display_name'], False), daemon=True))
            tr.start()

    def download_model(self, character, model, rvc):
        self.showLoaderPopup("Downloading Model", f"Downloading {character}")
        tr = (threading.Thread(target=falltalkutils.download_models, args=(self, character, model, rvc), daemon=True))
        tr.start()

    def update_model(self, character, model, rvc):
        self.showLoaderPopup("Updating Model", f"Updating {character}")
        folder = os.path.join("models", character, model['engine'])
        if os.path.exists(folder):
            shutil.rmtree(os.path.join("models", character, model['engine']))
        if rvc:
            rvc_folder = os.path.join("models", character, 'RVC')
            if os.path.exists(rvc_folder):
                shutil.rmtree(os.path.join("models", character, 'RVC'))
        tr = (threading.Thread(target=falltalkutils.download_models, args=(self, character, model, rvc), daemon=True))
        tr.start()

    def delete_model(self, character, model, display_name):
        title = f'Delete {display_name}'
        content = f"""
        Are you sure you would like to delete all {model['engine']} files for {display_name}? 
        This action cannot be undone, but you can download the files again at a later date.
        """
        w = MessageBox(title, content, self)
        w.yesButton.setText(self.tr('Yes'))
        if w.exec():
            shutil.rmtree(os.path.join("models", character, model['engine']))
            self.load_models_config()

    def delete_custom_model(self, character, display_name):
        title = f'Delete {display_name}'
        content = f"""
        Are you sure you would like to delete all files for {display_name}? 
        This action cannot be undone, but you can download the files again at a later date.
        """
        w = MessageBox(title, content, self)
        w.yesButton.setText(self.tr('Yes'))
        if w.exec():
            shutil.rmtree(os.path.join("models", character))
            self.load_models_config()


def hide_console():
    whnd = ctypes.windll.kernel32.GetConsoleWindow()
    if whnd != 0:
        ctypes.windll.user32.ShowWindow(whnd, 0)


if __name__ == '__main__':

    falltalkutils.seed_everything(0)

    if cfg.get(cfg.dpiScale) != "Auto":
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
        os.environ["QT_SCALE_FACTOR"] = str(cfg.get(cfg.dpiScale))

    application = QApplication(sys.argv)

    try:

        if cfg.get(cfg.huggingface_cache_dir) != "Please Select a Valid Folder":
            os.environ["HF_HUB_CACHE"] = cfg.get(cfg.huggingface_cache_dir)

        # These are needed for VoiceCraft
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['USER'] = getpass.getuser()
        os.environ['PHONEMIZER_ESPEAK_PATH'] = os.path.abspath(os.path.join("resource", "apps", "espeak", "espeak-ng.exe"))
        os.environ['ESPEAK_DATA_PATH'] = os.path.abspath(os.path.join("resource", "apps", "espeak", "espeak-ng-data"))
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = os.path.abspath(os.path.join("resource", "apps", "espeak", "libespeak-ng.dll"))
        # os.environ['FFMPEG_BIN'] = os.path.abspath(os.path.join("resource", "apps", "win_ffmpeg", "ffmpeg.exe"))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        falltalkutils.logger.debug('Unable to find espeak', e)

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.Ceil)

    app_id = 'falltalk'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)

    if cfg.get(cfg.first_start):
        cfg.set(cfg.themeMode, Theme.DARK)

    falltak_app = ModelApp()
    api_server = falltalkapi.FallTalkAPI(falltak_app)
    hide_console()
    application.exec()
    api_server.shutdown()
    sys.exit()
