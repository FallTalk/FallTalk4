from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import threading
import webbrowser
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tts_engines.whisper_engine import Whisper_Engine
    from tts_engines.apbwe_engine import APBWE_SR


import PySide6
from PySide6.QtCore import Qt, QSize, Slot, QUrl, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from packaging import version
from qfluentwidgets import FluentIcon as FIF, SplashScreen, StateToolTip, Flyout, InfoBarIcon, InfoBar, InfoBarPosition, \
    MessageBox
from qfluentwidgets import NavigationItemPosition

from src.config.config import cfg, DISCLAIMER, VERSION, RELEASE_URL
from src.enums.engine_type import EngineType
from src.tts_engines import tts_engine
from src.utils.audio_utils import combine_wav_files, combine_references
from src.utils.bulk_utils import bulk_inference, bulk_rvc_inference, bulk_fuz
from src.utils.file_utils import clean_folder, sanitize_filename, formatted_time_stamp_uuid, get_output_file_name
from src.utils.filesystem_utils import get_app_root
from src.utils.huggingface_utils import get_latest_release, get_model_diff, download_models, \
    download_all_models_config
from src.utils.icons import FallTalkIcons
from src.utils.inference_utils import (
    do_transcribe, eleven_labs_inference, edge_tts_inference, generic_inference,
    rvc_inference, do_transcribe_before_gen, preprocess_text, get_default_reference_and_transcript
)
from src.utils.logging_utils import logger
from src.utils.model_utils import (
    load_model, load_gpt_sovits, load_dia, load_rvc, load_spark,
    load_fish, load_f5, load_llasa, load_orpheus, load_style_tts2, load_upscaler, load_csm,
    load_higgs, load_chatterbox, load_dmo_speech2, load_vibe, load_qwen
)
# Import widgets here to avoid circular imports
from src.widgets import (
    UpscaleWidget, SettingsWidget, CharactersWidget, ReferencesWidget,
    FaqWidget, RVCWidget, BulkGenerationWidget,
    EzVoiceCreatorWidget, FallTalkWidget
)
from src.widgets.chat_widget import ChatWidget
from src.widgets.generic_generation_widget import GenericGenerationWidget
from src.widgets.falltalk_fluent_window import FallTalkFluentWindow
from src.widgets.generic_multi_generation_widget import GenericMultiGenerationWidget


class FallTalkApp(FallTalkFluentWindow):
    def __init__(self):
        super().__init__()
        self.stateTooltip = None
        self.icon_path = "resource/falltalk.png"  # Replace with the actual path to your icon file
        self.icon = QIcon(self.icon_path)
        self.splashScreen = SplashScreen(self.icon, self)
        self.splashScreen.setIconSize(QSize(128, 128))
        self.setupWindow()
        self.tts_engine: Optional['tts_engine'] = None
        self.transcription_engine: Optional['Whisper_Engine'] = None
        self.pending_character = None
        self.pending_rvc = None
        self.pending_model = None
        self.pending_base = False
        self.pending_bulk = False
        self.pending_ez = False
        self.characters_data = None
        self.models = None
        self.shared_models = None
        self.custom_models = None
        self.default_references = None
        self.upscale_engine = None
        self.apbwe_engine: Optional['APBWE_SR'] = None
        self.multi_generation_widget: Optional['GenericMultiGenerationWidget'] = None
        self.tts_widget: Optional['GenericGenerationWidget'] = None
        self.rvc_widget: Optional['RVCWidget'] = None
        self.bulk_generate_widget: Optional['BulkGenerationWidget'] = None
        self.faq_widget: Optional['FaqWidget'] = None
        self.characters_widget: Optional['CharactersWidget'] = None
        self.reference_widget: Optional['ReferencesWidget'] = None
        self.generate_widget: Optional['FallTalkWidget'] = None
        self.upscale_widget: Optional['UpscaleWidget'] = None
        self.setting_widget: Optional['SettingsWidget'] = None
        self.ez_voice_creator_widget: Optional['EzVoiceCreatorWidget'] = None
        self.chat_widget: Optional['ChatWidget'] = None


        # Dictionary to store engine actions
        self.engine_actions = {}
        # Dictionary to store engine load functions
        self.engine_load_functions = {
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
            EngineType.QWEN3_TTS: load_qwen
        }

        clean_folder("temp/")

        self.initUI()

        if cfg.get(cfg.download_configs) or cfg.get(cfg.first_start):
            self.download_models_config()

        if cfg.get(cfg.check_for_updates):
            self.checkForRelease()

        # Load static JSON data once at startup
        self._load_static_json_data()
        self.load_models_config()

        self.splashScreen.finish()

        if not cfg.get(cfg.accepted_disclaimer):
            self.verify()

        self.verifyFallout()

        if cfg.get(cfg.load_engine_art_start) and not cfg.get(cfg.first_start):
            self.onEngineChange(cfg.engine)

        if cfg.get(cfg.first_start):
            cfg.set(cfg.first_start, False)

        if cfg.get(cfg.api_only_mode):
            self.setVisible(False)

    def _load_static_json_data(self):
        """Load static JSON data from files and cache it in memory"""
        if self.characters_data is None:
            with open(os.path.join(get_app_root(), 'config/characters.json'), 'r', encoding='utf-8') as file:
                self.characters_data = {character['name']: character for character in json.load(file)}

        if self.models is None:
            # Load character-specific models from modelsv2.json
            with open(os.path.join(get_app_root(), 'config', 'modelsv2.json'), 'r', encoding='utf-8') as file:
                self.models = {model['name']: model for model in json.load(file)['characters']}

        if self.shared_models is None:
            # Load shared models from sharedmodels.json if it exists
            shared_models_path = os.path.join(get_app_root(), 'config', 'sharedmodels.json')
            if os.path.exists(shared_models_path):
                with open(shared_models_path, 'r', encoding='utf-8') as file:
                    self.shared_models = json.load(file)

                    # Process each shared model
                    for shared_model in self.shared_models:
                        engine = shared_model.get('engine')
                        engine_version = shared_model.get('engine_version', '1')
                        model_version = shared_model.get('model_version', '1')
                        model_name = shared_model.get('model_name')
                        model_type = shared_model.get('model_type', 'safetensors')
                        characters = shared_model.get('characters', [])

                        # Add the shared model to each character's models
                        for character_name in characters:
                            if character_name in self.models:
                                # If the character already exists, add the shared model to it
                                # if engine not in self.models[character_name]:
                                self.models[character_name][engine] = {
                                    'version': model_version,
                                    'engine_version': engine_version,
                                    'engine': engine,
                                    'type': model_type,
                                    'is_shared': True,
                                    'shared_model_name': model_name,
                                    'characters':  characters
                                }

        # Load default references from default_references.json if it exists
        default_references_path = os.path.join(get_app_root(), 'config', 'default_references.json')
        if os.path.exists(default_references_path):
            with open(default_references_path, 'r', encoding='utf-8') as file:
                self.default_references = json.load(file)
        else:
            self.default_references = None

    def _load_custom_models(self):
        """Load custom models from file - this can change during runtime"""
        if os.path.exists(os.path.join('config', 'custom_models.json')):
            with open(os.path.join(get_app_root(), 'config/custom_models.json'), 'r', encoding="utf-8") as file:
                self.custom_models = {model['name']: model for model in json.load(file)}
        else:
            self.custom_models = None

    def checkForRelease(self):
        try:
            latest = get_latest_release()
            my_version = version.parse(VERSION)
            latest_version = version.parse(latest)
            if latest_version.base_version > my_version.base_version:
                self.toolbar_1.addSeparator()
                self.toolbar_1.addAction(self.update_action)
                self.update_action.triggered.connect(lambda: webbrowser.open(RELEASE_URL))
        except Exception as e:
            logger.exception(f"Failed to fetch latest release: {e}")

    def setupWindow(self):
        self.setWindowTitle(f'FallTalk - {VERSION}')
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
        self.setMicaEffectEnabled(True)

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

    def download_models_config(self):
        try:
            old_json = None
            if os.path.exists(os.path.join(get_app_root(),'config/modelsv2.json')):
                with open(os.path.join(get_app_root(), 'config/modelsv2.json'), 'r', encoding="utf-8") as old_file:
                    old_json = json.load(old_file)

            download_all_models_config()

            with open(os.path.join(get_app_root(), 'config/modelsv2.json'), 'r', encoding="utf-8") as new_file:
                new_json = json.load(new_file)

            diff = get_model_diff(old_json, new_json)

            if diff is not None:
                self.toolbar_1.addSeparator()
                self.toolbar_1.addAction(self.new_models_action)
                w = MessageBox("Updates", diff, self)
                self.new_models_action.triggered.connect(lambda: w.open())

        except Exception as e:
            logger.exception("Unable to load configs")
            if not os.path.exists(os.path.join(get_app_root(), 'config/characters.json')) and not os.path.exists(os.path.join(get_app_root(), 'config/modelsv2.json')):
                self.createErrorInfoBar("Unable to Download Configs", "Unable to download the required configuration files, please check your network")

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
        # Initialize the two main engine widgets (RVG and non-RVG)
        self.rvc_widget = RVCWidget(self)
        self.tts_widget = GenericGenerationWidget(self, is_rvc=False)
        self.faq_widget = FaqWidget(self)
        self.characters_widget = CharactersWidget(self)
        self.reference_widget = ReferencesWidget(self)
        self.generate_widget = FallTalkWidget(parent=self, text="Generate Voice")
        self.bulk_generate_widget = BulkGenerationWidget(parent=self)
        self.upscale_widget = UpscaleWidget(parent=self)
        self.setting_widget = SettingsWidget(self)
        self.ez_voice_creator_widget = EzVoiceCreatorWidget(parent=self)
        self.chat_widget = ChatWidget(parent=self)
        self.multi_generation_widget = GenericMultiGenerationWidget(parent=self)

        # Add the two main engine widgets to the generate widget
        self.generate_widget.addToFrame(self.tts_widget)
        self.generate_widget.addToFrame(self.rvc_widget)

        # Connect RVC mic widget signal
        self.rvc_widget.rvc_mic_widget.media_recorder.doneRecording.connect(self.generate_audio)

        # Add navigation interfaces
        self.addSubInterface(self.characters_widget, FIF.PEOPLE, 'Character Models', NavigationItemPosition.TOP)
        self.addSubInterface(self.reference_widget, FIF.MIX_VOLUMES, 'Reference Audio', NavigationItemPosition.TOP)
        self.addSubInterface(self.generate_widget, FIF.ROBOT, 'Generation', NavigationItemPosition.TOP)
        self.addSubInterface(self.multi_generation_widget, FallTalkIcons.MULTI.icon(), 'Multi Generation', NavigationItemPosition.TOP)
        self.navigationInterface.addSeparator()
        self.addSubInterface(self.chat_widget, FIF.CHAT, 'Chat', NavigationItemPosition.TOP)
        self.navigationInterface.addSeparator()
        self.addSubInterface(self.bulk_generate_widget, FallTalkIcons.BULK.icon(), 'Bulk Generation', NavigationItemPosition.TOP)
        self.addSubInterface(self.upscale_widget, FallTalkIcons.UP.icon(stroke=True), 'Bulk Enhancement', NavigationItemPosition.TOP)
        self.addSubInterface(self.ez_voice_creator_widget, FallTalkIcons.MAGIC.icon(), 'ESP Voice Generator', NavigationItemPosition.TOP)
        self.navigationInterface.addSeparator()
        self.addSubInterface(self.faq_widget, FIF.HELP, 'FAQ', NavigationItemPosition.BOTTOM)
        self.addSubInterface(self.setting_widget, FIF.SETTING, 'Settings', NavigationItemPosition.BOTTOM)

        # Connect engine actions
        self.engine_actions[EngineType.RVC] = self.rvc_action
        self.engine_actions[EngineType.GPT_SOVITS] = self.gpt_sovits_action
        self.engine_actions[EngineType.STYLE_TTS2] = self.styletts2_action
        self.engine_actions[EngineType.FISH_SPEECH] = self.fish_action
        self.engine_actions[EngineType.F5] = self.f5_action
        self.engine_actions[EngineType.DIA] = self.dia_action
        self.engine_actions[EngineType.LLASA] = self.llasa_action
        self.engine_actions[EngineType.ORPHEUS] = self.orpheus_action
        self.engine_actions[EngineType.SPARK] = self.spark_action
        self.engine_actions[EngineType.CSM] = self.csm_action
        self.engine_actions[EngineType.HIGGS] = self.higgs_action
        self.engine_actions[EngineType.CHATTERBOX] = self.chatterbox_action
        self.engine_actions[EngineType.DMOSPEECH2] = self.dmo_speech2_action
        self.engine_actions[EngineType.VIBE] = self.vibe_action
        self.engine_actions[EngineType.QWEN3_TTS] = self.qwen_action

        # Connect action signals
        for engine_type, action in self.engine_actions.items():
            action.triggered.connect(lambda checked, et=engine_type: (
                cfg.set(cfg.engine, et.value),
                self.onEngineChange(et)
            ))

        # Connect device actions
        self.cpu_action.triggered.connect(self.cpu_checked)
        self.gpu_action.triggered.connect(self.gpu_checked)
        self.gpu2_action.triggered.connect(self.gpu2_checked)

        self.generate_widget.setEnabled(True)
        self.tts_widget.setEnabled(False)
        QApplication.processEvents()

    @Slot(PySide6.QtCore.QObject, str, str)
    def onError(self, parent: 'FallTalkApp', title, text):
        parent.complete_loader()
        parent.createErrorInfoBar(title, text)

    @Slot(PySide6.QtCore.QObject, str, str)
    def onWarn(self, parent: 'FallTalkApp', title, text):
        parent.createErrorInfoBar(title, text)

    @Slot( str, str)
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

    @Slot(str, str)
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

    @Slot(PySide6.QtCore.QObject)
    def close_loader(self, parent: 'FallTalkApp'):
        parent.complete_loader()

    def complete_loader(self):
        if self.stateTooltip:
            self.stateTooltip.setContent("Completed")
            self.stateTooltip.setState(True)
            self.stateTooltip = None

        self.setEnabled(True)

    def onDeviceChange(self):
        logger.debug(f'Device Changed {cfg.get(cfg.device)}')
        # tr = (threading.Thread(target=self.clean_engines, daemon=True))
        # tr.start()

        if self.tts_engine is not None:
            self.onEngineChange(cfg.get(cfg.engine))

    @Slot(str)
    def engine_change(self, engine):
        cfg.set(cfg.engine, engine)
        self.onEngineChange(cfg.get(cfg.engine))

    def onEngineChange(self, engine):
        self.showLoaderPopup("Loading Engine", f"Loading {engine.value}")
        logger.debug(f"Engine Changed to {engine.value}")
        engine_type = EngineType(engine.value)
        
        # Hide both main widgets initially
        self.tts_widget.setVisible(False)
        self.tts_widget.setEnabled(False)
        self.rvc_widget.setVisible(False)
        self.rvc_widget.setEnabled(False)

        self.chat_widget.clear_chat()
        self.chat_widget.setEnabled(False)
        self.multi_generation_widget.clear_results()
        self.multi_generation_widget.setEnabled(False)
        self.bulk_generate_widget.setEnabled(False)
        self.character_label.setText(f"Please Load Model")

        # Uncheck all engine actions
        for action in self.engine_actions.values():
            action.setChecked(False)

        if self.tts_engine is not None:
            self.tts_engine.clean()

        # Update the widget based on engine type
        if engine_type == EngineType.RVC:
            self.rvc_widget.setVisible(True)
            self.rvc_widget.setEnabled(True)
            self.engine_actions[engine_type].setChecked(True)
        else:
            # For all other engines, update the TTS widget
            self.tts_widget.change_engine(engine_type)
            self.tts_widget.setVisible(True)
            self.engine_actions[engine_type].setChecked(True)
            # Also update the multi-generation widget
            self.multi_generation_widget.update_engine_type(engine_type)
            self.chat_widget.update_engine_type(engine_type)
            self.ez_voice_creator_widget.update_engine_type(engine_type)

        if engine_type in self.engine_load_functions:
            tr = (threading.Thread(target=self.engine_load_functions[engine_type], args={self}, daemon=True))
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

    @Slot(PySide6.QtCore.QObject, str)
    def after_engine_load(self, parent, engine):
        engine_type = EngineType(engine)
        parent.complete_loader()
        
        # Enable the appropriate widget based on engine type
        if engine_type == EngineType.RVC:
            widget = self.rvc_widget
        else:
            widget = self.tts_widget
            
        widget.setEnabled(False)
        widget.setVisible(True)
        widget.media_player.setVisible(True)
        parent.bulk_generate_widget.setEnabled(True)

    @Slot(PySide6.QtCore.QObject)
    def afterModelLoader(self, parent: 'FallTalkApp'):
        parent.complete_loader()
        engine_type = EngineType(cfg.get(cfg.engine))
        self.chat_widget.clear_chat()
        self.multi_generation_widget.clear_results()

        # Get the appropriate widget based on engine type
        if engine_type == EngineType.RVC:
            widget = self.rvc_widget
            self.multi_generation_widget.setEnabled(False)
            self.chat_widget.setEnabled(False)
        else:
            widget = self.tts_widget
            self.multi_generation_widget.setEnabled(True)
            self.chat_widget.setEnabled(True)

        widget.setEnabled(True)
        if engine_type == EngineType.RVC:
            self.stackedWidget.setCurrentWidget(self.generate_widget)
        elif (self.tts_engine.is_base or engine_type.needs_reference_when_trained) and engine_type.needs_transcription:
            widget.text_input.setPlaceholderText(
                "If no reference is selected, a default will be used. Selecting a reference audio can help change the emotion of the generated speech. ")
            # widget.transcribe_button.setVisible(True)
            self.stackedWidget.setCurrentWidget(self.reference_widget)
        elif self.tts_engine.is_base or engine_type.needs_reference_when_trained:
            widget.text_input.setPlaceholderText(
                "If no reference is selected, a default will be used. Selecting a reference audio can help change the emotion of the generated speech. ")
            # widget.transcribe_button.setVisible(False)
            self.stackedWidget.setCurrentWidget(self.reference_widget)
        else:
            widget.text_input.setPlaceholderText("Please Enter Text. This model does not need a reference, however selecting one can increase quality.")
            # widget.transcribe_button.setVisible(False)
            widget.generate_button.setEnabled(True)
            self.stackedWidget.setCurrentWidget(self.generate_widget)


    @Slot(PySide6.QtCore.QObject)
    def afterGen(self, parent: 'FallTalkApp'):
        parent.complete_loader()

    @Slot(PySide6.QtCore.QObject)
    def afterDownload(self, parent: 'FallTalkApp'):
        parent.complete_loader()

    @Slot(PySide6.QtCore.QObject)
    def afterModelDownload(self, parent: 'FallTalkApp'):
        parent.complete_loader()
        parent.load_models_config()

    @Slot(PySide6.QtCore.QObject)
    def afterModelConfigLoad(self, parent: 'FallTalkApp'):
        parent.complete_loader()

    @Slot(PySide6.QtCore.QObject)
    def continueLoad(self, parent: 'FallTalkApp'):
        self.bulk_generate_widget.setEnabled(True)
        parent.load_models_config()
        if parent.pending_bulk:
            parent.bulk_inference()
        elif parent.pending_ez:
            parent.ez_voice_creator_inference(cfg.get(cfg.ez_total))
        elif parent.pending_character and not parent.pending_base:
            parent.stackedWidget.setCurrentWidget(parent.generate_widget)
            parent.load_trained_model(parent.pending_character, parent.pending_model, parent.pending_rvc)
        elif parent.pending_base:
            parent.stackedWidget.setCurrentWidget(parent.generate_widget)
            parent.load_base_model(parent.pending_character, parent.pending_rvc)
        else:
            parent.stackedWidget.setCurrentWidget(parent.characters_widget)


    def find_first_match_by_name(self, models, name):
        for character in models['characters']:
            if character['name'] == name:
                return character
        return None

    def load_models_config(self):
        """Load models configuration using cached JSON data"""
        # Reload custom models since they can change during runtime
        self._load_custom_models()

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
                    model = self.models[character_name][cfg.get(cfg.engine)]
                    engine_type = EngineType(cfg.get(cfg.engine))
                    version = model.get('engine_version', '1')

                    # Validate version compatibility
                    if engine_type.is_version_supported(version):
                        model_found = True
                        c = copy.copy(character)
                        c['display_name'] = self.models[character_name]['display_name']
                        c[cfg.get(cfg.engine)] = model
                        trained_characters.append(c)
                    else:
                        logger.warning(f"Skipping {character_name} - Engine version {version} not supported for {cfg.get(cfg.engine)}")

            if not model_found:
                if 'display_name' not in character:
                    character['display_name'] = character['name']
                untrained_characters.append(character)

        self.characters_widget.clear()
        self.characters_widget.loadTrained(self, trained_characters)
        if cfg.get(cfg.engine) != 'RVC':
            self.characters_widget.loadUntrained(self, untrained_characters)

        if self.custom_models is not None:
            # Filter custom models to only include those with the current engine and supported versions
            filtered_custom_models = {}
            for name, model in self.custom_models.items():
                if cfg.get(cfg.engine) in model:
                    engine_type = EngineType(cfg.get(cfg.engine))
                    version = model.get('engine_version', '1')
                    if engine_type.is_version_supported(version):
                        filtered_custom_models[name] = model
                    else:
                        logger.warning(f"Skipping custom model {name} - Engine version {version} not supported for {cfg.get(cfg.engine)}")

            self.characters_widget.loadCustom(self, filtered_custom_models)

        self.bulk_generate_widget.populate_character_card()

    def update_reference_table(self, selected_model):
        if selected_model:
            self.reference_widget.addDataToReferencesTable(selected_model)

    def transcribe(self, widget):
        references = self.reference_widget.reference_audio
        if references is None or not references:
            self.showErrorPopup(widget, widget.transcribe_button, "Please Select Reference Audio")
            self.complete_loader()
        else:
            if len(references) == 1:
                logger.debug(references)
                selected_audio = references[0]
            else:
                logger.debug(references)
                selected_audio = combine_wav_files(references)

            self.showLoaderPopup("Transcribing Audio", "Please Wait")
            tr = (threading.Thread(target=do_transcribe, args=(self, selected_audio, widget), daemon=True))
            tr.start()

    def transcribe_before_gen(self, references):
            # Check if we're using F5 in edit mode
            current_engine = EngineType(cfg.get(cfg.engine))
            is_f5_edit_mode = current_engine == EngineType.F5 and cfg.get(cfg.f5_mode) == "edit"

            # Check if we have stored transcripts from the references widget
            combined_transcript = self.reference_widget.get_combined_transcript()

            # If we have a combined transcript and we're not in F5 edit mode, use it directly
            if combined_transcript and not is_f5_edit_mode:
                # Create a transcribe state with the combined transcript
                transcribe_state = {'transcript': combined_transcript}
                # Skip transcription and go directly to generation
                self.generate_audio(None, transcribe_state)
                return

            # Otherwise, proceed with transcription as before
            if len(references) == 1:
                selected_audio = references[0]
            else:
                selected_audio = combine_wav_files(references)

            self.showLoaderPopup("Transcribing Audio", "Please Wait")
            tr = (threading.Thread(target=do_transcribe_before_gen, args=(self, selected_audio), daemon=True))
            tr.start()

    @Slot(PySide6.QtCore.QObject, PySide6.QtCore.QObject)
    def after_transcribe(self, parent: 'FallTalkApp', widget):
        widget.clear()
        widget.load_data()
        parent.complete_loader()

    @Slot(PySide6.QtCore.QObject, str)
    def after_transcribe_gen(self, parent: 'FallTalkApp', transcribe_state):
        parent.complete_loader()
        t = json.loads(transcribe_state)
        parent.generate_audio(None, t)

    def get_music_file(self, file_name, wav=False):
        if file_name is None or file_name == "" or file_name == "Random":
            file_name = formatted_time_stamp_uuid()

        file_name = sanitize_filename(file_name)
        path = os.path.abspath(os.path.join(cfg.get(cfg.output_dir), "music", f"{file_name}.wav" if wav else f"{file_name}"))
        os.makedirs(os.path.join(cfg.get(cfg.output_dir), "music"), exist_ok=True)
        return path

    def get_fx_file(self, file_name, wav=False):
        if file_name is None or file_name == "" or file_name == "Random":
            file_name = formatted_time_stamp_uuid()

        file_name = sanitize_filename(file_name)
        path = os.path.abspath(os.path.join(cfg.get(cfg.output_dir), "fx", f"{file_name}.wav" if wav else f"{file_name}"))
        os.makedirs(os.path.join(cfg.get(cfg.output_dir), "fx"), exist_ok=True)
        return path

    def get_output_file(self, widget):
        return get_output_file_name(widget.output_name.value, cfg.get(cfg.output_dir), self.tts_engine.model_name, self.tts_engine.engine_type.value)


    @Slot(PySide6.QtCore.QObject)
    def after_upscale(self, parent: 'FallTalkApp'):
        parent.complete_loader()
        parent.upscale_folder()

    def upscale_folder(self):
        if self.upscale_engine is None:
            self.showLoaderPopup("Loading Enhancer", "Please Wait")
            tr = (threading.Thread(target=load_upscaler, args={self}, daemon=True))
            tr.start()
        else:
            up_dir = self.upscale_widget.upscale_dir.value
            if up_dir is None or up_dir == "Please Select a Valid Folder":
                self.showErrorPopup(self.upscale_widget, self.upscale_widget.generate_button, "Please Select a Valid Folder")
            else:
                self.showLoaderPopup("Enhancing", "Please Wait")
                tr = (threading.Thread(target=self.upscale_engine.upscale_dir, args=(up_dir, cfg.get(cfg.replace_existing), cfg.get(cfg.include_subdir), self.upscale_widget.audio_mode.value, self.upscale_widget.sample_rate.value), daemon=True))
                tr.start()

    def ez_voice_creator_inference(self, runs=1):
        """Handle generation button click in EzVoiceCreator widget"""
        if not self.ez_voice_creator_widget.dialogue_table.model() or self.ez_voice_creator_widget.dialogue_table.model().rowCount() == 0:
            MessageBox("Error", "No dialogue data to generate", self).exec()
            return

        if self.tts_engine is None:
            self.pending_ez = True
            self.onEngineChange(cfg.engine)
        else:
            self.pending_ez = False
            data = len(self.ez_voice_creator_widget.dialogue_table.model().getData())
            if data > 0:
                self.showLoaderPopup(f"Generating Audio", f"Completed: 0/{data}")
                from src.utils.bulk_utils import ez_voice_creator_inference
                tr = (threading.Thread(target=ez_voice_creator_inference, args=(self, runs), daemon=True))
                tr.start()
            else:
                self.showErrorPopup(self.ez_voice_creator_widget, self.ez_voice_creator_widget.generate_button, "Please Load some Data")

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
                    tr = (threading.Thread(target=bulk_inference, args={self}, daemon=True))
                    tr.start()
                else:
                    self.showErrorPopup(self.bulk_generate_widget, self.bulk_generate_widget.stackedWidget.currentWidget().generate_button, "Please Select Load some Data")
            elif self.bulk_generate_widget.stackedWidget.currentWidget() == self.bulk_generate_widget.bulk_rvc_widget:
                rvc_dir = self.bulk_generate_widget.bulk_rvc_widget.rvc_dir.value
                if rvc_dir is None or rvc_dir == "Please Select a Valid Folder":
                    self.showErrorPopup(self.bulk_generate_widget, self.bulk_generate_widget.stackedWidget.currentWidget().generate_button, "Please Select a Valid Folder")
                elif self.bulk_generate_widget.bulk_rvc_widget.character_card.configItem.text() is None or self.bulk_generate_widget.bulk_rvc_widget.character_card.configItem.text() == '':
                    self.showErrorPopup(self.bulk_generate_widget, self.bulk_generate_widget.stackedWidget.currentWidget().generate_button, "Please Select a valid Character")
                else:
                    self.showLoaderPopup(f"Generating Bulk RVC Audio", f"Gathering Files")
                    tr = (threading.Thread(target=bulk_rvc_inference, args=(self, rvc_dir, self.bulk_generate_widget.bulk_rvc_widget.character_card.configItem.currentData(), cfg.get(cfg.include_subdir), cfg.get(cfg.replace_existing), cfg.get(cfg.threads), cfg.get(cfg.use_existing_lip)), daemon=True))
                    tr.start()
            elif self.bulk_generate_widget.stackedWidget.currentWidget() == self.bulk_generate_widget.bulk_fuz_widget:
                lip_dir = self.bulk_generate_widget.bulk_fuz_widget.lip_dir.value
                if lip_dir is None or lip_dir == "Please Select a Valid Folder":
                    self.showErrorPopup(self.bulk_generate_widget, self.bulk_generate_widget.stackedWidget.currentWidget().generate_button, "Please Select a Valid Folder")
                else:
                    self.showLoaderPopup(f"Generating Bulk FUZ Audio", f"Gathering Files")
                    tr = (threading.Thread(target=bulk_fuz, args=(self, lip_dir, cfg.get(cfg.include_subdir), cfg.get(cfg.threads), cfg.get(cfg.use_existing_lip)), daemon=True))
                    tr.start()

    def generate_audio(self, recording_file=None, transcribe_state=None):
        references = self.reference_widget.reference_audio
        references_length = self.reference_widget.reference_audio_length
        current_engine = EngineType(cfg.get(cfg.engine))
        
        # Get the appropriate widget based on engine type
        if current_engine == EngineType.RVC:
            current_widget = self.rvc_widget
        else:
            current_widget = self.tts_widget

        if current_engine == EngineType.RVC:
            if current_widget.stackedWidget.currentWidget() == current_widget.rvc_mic_widget:
                if recording_file is None or not recording_file:
                    self.showErrorPopup(current_widget, current_widget.rvc_mic_widget.media_recorder.recordButton, "Please Record Audio")
                elif self.tts_engine.model_name is None:
                    self.showErrorPopup(current_widget, current_widget.rvc_mic_widget.media_recorder.recordButton, "Please Select Load a Model")
                else:
                    self.showLoaderPopup("Cloning Input Audio", "Please Wait")
                    output_file = self.get_output_file(current_widget.rvc_mic_widget)
                    shutil.copy(recording_file, output_file)
                    tr = (threading.Thread(target=rvc_inference, args=(self, output_file, current_widget), daemon=True))
                    tr.start()
            elif current_widget.stackedWidget.currentWidget() == current_widget.rvc_file_widget:
                recording_file = current_widget.rvc_file_widget.rvc_file.value
                if recording_file is None:
                    self.showErrorPopup(current_widget, current_widget.rvc_file_widget, "Please Select a file")
                elif self.tts_engine.model_name is None:
                    self.showErrorPopup(current_widget, current_widget.rvc_file_widget, "Please Select Load a Model")
                else:
                    self.showLoaderPopup("Cloning File Audio", "Please Wait")
                    output_file = self.get_output_file(current_widget.rvc_file_widget)
                    shutil.copy(recording_file, output_file)
                    tr = (threading.Thread(target=rvc_inference, args=(self, output_file, current_widget), daemon=True))
                    tr.start()
            elif current_widget.stackedWidget.currentWidget() == current_widget.edge_tts_widget:
                text = preprocess_text(current_widget.edge_tts_widget.text_input.toPlainText())
                if not text or text == '':
                    self.showErrorPopup(current_widget, current_widget.edge_tts_widget.generate_button, "Please Enter Some Text to Generate...")
                elif self.tts_engine.model_name is None:
                    self.showErrorPopup(current_widget, current_widget.edge_tts_widget.generate_button, "Please Select Load a Model")
                else:
                    self.showLoaderPopup("Cloning TTS from Edge", "Please Wait")
                    output_file = self.get_output_file(current_widget.edge_tts_widget)
                    tr = (threading.Thread(target=edge_tts_inference, args=(self, text, output_file, current_widget.edge_tts_widget.voice_combo.configItem.currentText(), current_widget), daemon=True))
                    tr.start()
            elif current_widget.stackedWidget.currentWidget() == current_widget.eleven_labs_widget:
                text = preprocess_text(current_widget.eleven_labs_widget.text_input.toPlainText())
                if not text or text == '':
                    self.showErrorPopup(current_widget, current_widget.eleven_labs_widget.generate_button, "Please Enter Some Text to Generate...")
                elif self.tts_engine.model_name is None:
                    self.showErrorPopup(current_widget, current_widget.eleven_labs_widget.generate_button, "Please Select Load a Model")
                else:
                    self.showLoaderPopup("Calling ElevenLabs", "Please Wait")
                    output_file = self.get_output_file(current_widget.eleven_labs_widget)
                    tr = (threading.Thread(target=eleven_labs_inference, args=(self, text, output_file, current_widget.eleven_labs_widget.voice_combo.configItem.currentText(), current_widget), daemon=True))
                    tr.start()
        else:

            if not current_widget.text_input.toPlainText():
                self.showErrorPopup(current_widget, current_widget.generate_button, "Please Enter Some Text")
                return
            else:
                text = preprocess_text(current_widget.text_input.toPlainText())

            if current_engine.needs_reference_when_trained or self.tts_engine.is_base:
                if references is None or not references:
                    # Get default reference from default_references.json and characters.json
                    character_name = self.tts_engine.model_name
                    reference_path, transcribe_state = get_default_reference_and_transcript(self, character_name)

                    # If reference found, use it
                    if reference_path:
                        references = [reference_path]
                        # Estimate reference length (5 seconds is a reasonable default)
                        references_length = 5
                    else:
                        self.showErrorPopup(current_widget, current_widget.generate_button, "Please Select Reference Audio")
                        return
                elif references_length > current_engine.max_reference_length or references_length < 3:
                    self.showErrorPopup(current_widget, current_widget.generate_button, f"Please Select between 3 and {current_engine.max_reference_length} seconds of Reference Audio")
                    return
                elif references_length < 3:
                    self.showErrorPopup(current_widget, current_widget.generate_button, "Please Select at least 3 seconds of Reference Audio")
                    return

            # Only transcribe if we don't already have a transcript from default references
            if references and transcribe_state is None:
                self.transcribe_before_gen(references)
                return

            self.showLoaderPopup("Generating Audio", "Please Wait")
            # F5 has special handling with start_time and end_time parameters
            if current_engine == EngineType.F5:
                start_word = current_widget.start_dropdown_card.getWordInfo()
                end_word = current_widget.end_dropdown_card.getWordInfo()
                start_time = start_word['start'] if start_word is not None else None
                end_time = end_word['end'] if end_word is not None else None
                if cfg.get(cfg.f5_mode) == "edit" and start_time is None and end_time is None:
                    self.showErrorPopup(current_widget, current_widget.generate_button, "Please Select Words to Edit")
                elif cfg.get(cfg.f5_mode) == "edit" and start_time > end_time:
                    self.showErrorPopup(current_widget, current_widget.generate_button, "Start must come before the end time")
                else:
                    threading.Thread(
                        target=generic_inference,
                        args=(
                            self,
                            self.get_output_file(current_widget),
                            text,
                            combine_references(references),
                            current_widget,
                            transcribe_state,
                        ),
                        kwargs={
                            'start_time': start_time,
                            'end_time': end_time,
                            'speaker': self.tts_engine.model_name,
                            'api': False
                        },
                        daemon=True
                    ).start()
            else:
                # All other engines use generic_inference
                threading.Thread(
                    target=generic_inference,
                    args=(
                        self,
                        self.get_output_file(current_widget),
                        text,
                        combine_references(references),
                        current_widget,
                        transcribe_state,
                    ),
                    kwargs={
                        'speaker': self.tts_engine.model_name,
                        'api': False
                    },
                    daemon=True
                ).start()


    def showErrorPopup(self, parent, target, content):
        Flyout.create(
            icon=InfoBarIcon.ERROR,
            title='Error',
            content=content,
            target=target,
            parent=parent,
            isClosable=True
        )

    def load_custom_model(self, character, model, rvc):
        if cfg.get(cfg.engine) in model:
            self.load_trained_model(character, model, rvc)
        else:
            self.load_base_model(character, model, rvc)

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
            # Enable RVC for the TTS widget if RVC is available
            if hasattr(self.tts_widget, 'rvc_enabled'):
                self.tts_widget.rvc_enabled.setVisible(rvc is not None)

            self.showLoaderPopup("Loading Base Model", f"Loading")
            tr = (threading.Thread(target=load_model, args=(self, character['name'], rvc, character['display_name'], base_model), daemon=True))
            tr.start()

    def load_trained_model(self, character, model, rvc):
        if self.tts_engine is None and cfg.engine is not None:
            self.pending_character = character
            self.pending_model = model
            self.pending_rvc = rvc
            self.pending_base = False
            self.onEngineChange(cfg.engine)
        else:
            if model:
                # Validate engine version
                engine_type = EngineType(cfg.get(cfg.engine))
                version = model[engine_type.value].get('engine_version', "1")
                if not engine_type.is_version_supported(version):
                    self.showErrorPopup(self, self.characters_widget, f"Engine version {version} is not supported for {cfg.get(cfg.engine)}")
                    return

                if cfg.get(cfg.fallout_4_directory) == "fallout4.exe not found":
                    self.verifyFallout()
                else:
                    self.reference_widget.addDataToReferencesTable(model)

                self.update_reference_table(model)
                self.pending_character = None
                self.pending_model = None
                self.pending_rvc = None

                self.tts_widget.rvc_enabled.setVisible(rvc is not None)
                self.multi_generation_widget.rvc_enabled.setVisible(rvc is not None)
                self.chat_widget.rvc_enabled.setVisible(rvc is not None)

                self.showLoaderPopup("Loading Model", f"Loading {character}")
                tr = (threading.Thread(target=load_model, args=(self, character, rvc, model['display_name'], False, version), daemon=True))
                tr.start()

    def download_model(self, character, model, rvc):
        self.showLoaderPopup("Downloading Model", f"Downloading {character}")
        tr = (threading.Thread(target=download_models, args=(self, character, model, rvc), daemon=True))
        tr.start()

    def download_model_and_load(self, character, model, rvc):
        self.showLoaderPopup("Downloading Model", f"Downloading {character}")
        tr = (threading.Thread(target=download_models, args=(self, character, model, rvc), daemon=True))
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
        tr = (threading.Thread(target=download_models, args=(self, character, model, rvc), daemon=True))
        tr.start()

    def delete_model(self, character, model, display_name):

        is_shared = model.get('is_shared', False)
        shared_model_name = model.get('shared_model_name')

        if is_shared:
            title = f'Delete Shared Model {shared_model_name}'
            content = f"""
            Are you sure you would like to delete all {model['engine']} files for {shared_model_name}? 
            This action cannot be undone, but you can download the files again at a later date.
            This is a shared model, deleting it will remove the model for all characters on the model.
            """
        else:
            title = f'Delete {display_name}'
            content = f"""
            Are you sure you would like to delete all {model['engine']} files for {display_name}? 
            This action cannot be undone, but you can download the files again at a later date.
            """
        w = MessageBox(title, content, self)
        w.yesButton.setText(self.tr('Yes'))
        if w.exec():
            engine_type = EngineType(model['engine'])
            model_engine_version = model.get('engine_version', 1)

            version = model.get('version', 1)

            if is_shared and shared_model_name:
                shutil.rmtree(os.path.join(get_app_root(), "models", "shared", engine_type.get_model_path(shared_model_name, model_engine_version)))
            else:
                shutil.rmtree(os.path.join(get_app_root(), "models", engine_type.get_model_path(character, version)))

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
            if os.path.exists(os.path.join("models", character)):
                shutil.rmtree(os.path.join("models", character))

            if os.path.exists(os.path.join('config', 'custom_models.json')):
                with open(os.path.join('config', 'custom_models.json'), 'r', encoding="utf-8") as file:
                    custom_models = json.load(file)
            else:
                custom_models = []

            new_custom_models = []
            for model in custom_models:
                if model['display_name'] != display_name:
                    new_custom_models.append(model)

            with open(os.path.join('config', 'custom_models.json'), 'w', encoding="utf-8") as file:
                json.dump(new_custom_models, file)

            self.load_models_config()
