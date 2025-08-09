from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Type

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

from PySide6.QtWidgets import QGroupBox, QHBoxLayout
from qfluentwidgets import FluentIcon as FIF, PrimaryPushButton
from src.ui.cards import RadioSettingCard, ComboBoxWordsCard

from audio.audio_player import StandardAudioPlayerBar
from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.utils.logging_utils import logger
from src.widgets.generation_widget import GenerationWidget

# Import all settings and help widgets
from src.settings.fish_speech_settings import FishSpeechSettings
from src.settings.gpt_sovits_settings import GPTSoVITSSettings
from src.settings.llasa_settings import LlasaSettings
from src.settings.higgs_settings import HiggsSettings
from src.settings.spark_settings import SparkSettings
from src.settings.orpheus_settings import OrpheusSettings
from src.settings.dia_settings import DIASettings
from src.settings.csm_settings import CSMSettings
from src.settings.chatterbox_settings import ChatterboxSettings
from src.settings.xtts_settings import XTTSSettings
from src.settings.styletts2_settings import StyleTTS2Settings
from src.settings.f5_settings import F5Settings
from src.settings.dmo_speech2_settings import DMSpeech2Settings

from src.help.fish_help import FishHelp
from src.help.gpt_sovits_help import GPTSoVITSHelp
from src.help.llasa_help import LlasaHelp
from src.help.higgs_help import HiggsHelp
from src.help.spark_help import SparkHelp
from src.help.orpheus_help import OrpheusHelp
from src.help.dia_help import DIAHelp
from src.help.csm_help import CSMHelp
from src.help.chatterbox_help import ChatterboxHelp
from src.help.xtts_help import XTTSHelp
from src.help.styletts2_help import StyleTTS2Help
from src.help.f5_help import F5Help
from src.help.dmo_speech2_help import DMSpeech2Help

class GenericGenerationWidget(GenerationWidget):
    """
    A generic widget that combines functionality from all generation widgets.
    It uses the engine type to determine its behavior and appearance.
    """

    # Mapping of engine types to their settings and help widgets
    SETTINGS_WIDGETS: Dict[EngineType, Type] = {
        EngineType.FISH_SPEECH: FishSpeechSettings,
        EngineType.GPT_SOVITS: GPTSoVITSSettings,
        EngineType.LLASA: LlasaSettings,
        EngineType.HIGGS: HiggsSettings,
        EngineType.SPARK: SparkSettings,
        EngineType.ORPHEUS: OrpheusSettings,
        EngineType.DIA: DIASettings,
        EngineType.CSM: CSMSettings,
        EngineType.CHATTERBOX: ChatterboxSettings,
        EngineType.XTTS_V2: XTTSSettings,
        EngineType.STYLE_TTS2: StyleTTS2Settings,
        EngineType.F5: F5Settings,
        EngineType.DMOSPEECH2: DMSpeech2Settings,
    }

    HELP_WIDGETS: Dict[EngineType, Type] = {
        EngineType.FISH_SPEECH: FishHelp,
        EngineType.GPT_SOVITS: GPTSoVITSHelp,
        EngineType.LLASA: LlasaHelp,
        EngineType.HIGGS: HiggsHelp,
        EngineType.SPARK: SparkHelp,
        EngineType.ORPHEUS: OrpheusHelp,
        EngineType.DIA: DIAHelp,
        EngineType.CSM: CSMHelp,
        EngineType.CHATTERBOX: ChatterboxHelp,
        EngineType.XTTS_V2: XTTSHelp,
        EngineType.STYLE_TTS2: StyleTTS2Help,
        EngineType.F5: F5Help,
        EngineType.DMOSPEECH2: DMSpeech2Help,
    }

    def __init__(self, parent: FallTalkApp, engine_type: EngineType):
        super().__init__(parent=parent, text=engine_type.value)
        self.engine_type = engine_type
        self.text_input.setPlaceholderText("If no reference is selected, a default will be used. Selecting a reference audio can help change the emotion of the generated speech. ")
        self.transcribe_state = None
        self.words_data = None

        # F5-specific UI elements
        self.start_dropdown_card = None
        self.end_dropdown_card = None
        self.start_and_end = None
        self.temp_and_rep = None
        self.transcribe_button = None

        # Add F5-specific UI elements if this is an F5 widget
        if engine_type == EngineType.F5:
            self.setup_f5_specific_ui()

        self.addGenSettings()

        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVolume(100)
        self.buttons_layout = QHBoxLayout()

        # Add transcribe button for F5 in edit mode
        if engine_type == EngineType.F5:
            self.transcribe_button = PrimaryPushButton(text="Transcribe Reference Audio")
            self.transcribe_button.setIcon(FIF.PENCIL_INK)
            self.transcribe_button.clicked.connect(self.transcribe)
            self.transcribe_button.setEnabled(False)
            self.transcribe_button.setVisible(cfg.get(cfg.f5_mode) == "edit")
            self.buttons_layout.addWidget(self.transcribe_button, stretch=1)

        # Add generate button
        self.generate_button = PrimaryPushButton(text="Generate Audio")
        self.generate_button.setIcon(FIF.SEND)

        # Special case for F5 to pass transcribe_state
        if engine_type == EngineType.F5:
            self.generate_button.clicked.connect(lambda: self.parent.generate_audio(transcribe_state=self.transcribe_state))
        else:
            self.generate_button.clicked.connect(self.parent.generate_audio)

        self.buttons_layout.addWidget(self.generate_button, stretch=1)

        # Add settings and help widgets based on engine type
        if engine_type in self.SETTINGS_WIDGETS:
            self.settings_drawer.addWidget(self.SETTINGS_WIDGETS[engine_type](self))

        if engine_type in self.HELP_WIDGETS:
            self.help_drawer.addWidget(self.HELP_WIDGETS[engine_type](self))

        self.buttons_layout.addWidget(self.settings_button)
        self.buttons_layout.addWidget(self.help_button)

        self.boxLayout.addLayout(self.buttons_layout)
        self.addToFrame(self.media_player)

        self.setVisible(cfg.engine.value == engine_type.value)
        self.media_player.setVisible(cfg.engine.value == engine_type.value)
        self.setEnabled(False)

    def setup_f5_specific_ui(self):
        """Set up the F5-specific UI elements for edit mode"""
        self.mode_card = RadioSettingCard(
            cfg.f5_mode,
            FIF.DEVELOPER_TOOLS,
            self.tr('Generation Mode'),
            self.tr('How should we generate text'),
            texts=["Edit", "TTS"],
        )

        self.mode_card.optionChanged.connect(self.mode_changed)
        self.addToFrame(self.mode_card)

        self.start_dropdown_card = ComboBoxWordsCard(
            FIF.RIGHT_ARROW,
            self.tr('Start'),
            self.tr('Where do we start generating the new text'))
        self.end_dropdown_card = ComboBoxWordsCard(
            FIF.LEFT_ARROW,
            self.tr('End'),
            self.tr('Where do we stop generating the new text'))

        self.start_and_end = QGroupBox()
        self.start_and_end.setStyleSheet("border: none")
        self.start_and_end_layout = QHBoxLayout()
        self.start_and_end_layout.setContentsMargins(0, 0, 0, 0)
        self.start_and_end_layout.addWidget(self.start_dropdown_card, 3)
        self.start_and_end_layout.addWidget(self.end_dropdown_card, 3)
        self.start_and_end.setLayout(self.start_and_end_layout)
        self.start_and_end.setVisible(cfg.get(cfg.f5_mode) == "edit")
        self.addToFrame(self.start_and_end)

        self.temp_and_rep = QGroupBox()
        self.temp_and_rep.setStyleSheet("border: none")
        self.temp_and_rep_layout = QHBoxLayout()
        self.temp_and_rep_layout.setContentsMargins(0, 0, 0, 0)
        self.temp_and_rep.setLayout(self.temp_and_rep_layout)
        self.temp_and_rep.setVisible(cfg.get(cfg.f5_mode) == "edit")
        self.addToFrame(self.temp_and_rep)

    def mode_changed(self, change):
        """Handle mode changes for F5 widget"""
        if self.engine_type != EngineType.F5:
            return

        self.start_and_end.setVisible(change.value == "edit")
        self.temp_and_rep.setVisible(change.value == "edit")
        self.transcribe_button.setVisible(change.value == "edit")

    def transcribe(self):
        """Transcribe reference audio"""
        self.parent.transcribe(self)

    def onReferenceSelect(self):
        """Handle reference selection"""
        if self.engine_type == EngineType.F5:
            self.generate_button.setEnabled(True)
            if self.transcribe_button:
                self.transcribe_button.setEnabled(True)
        else:
            if self.parent.tts_engine and self.parent.tts_engine.is_base:
                self.generate_button.setEnabled(True)
            elif self.parent.tts_engine:
                self.generate_button.setEnabled(True)

    def clear(self):
        """Clear the widget state"""
        if self.engine_type == EngineType.F5:
            if self.transcribe_button:
                self.transcribe_button.setEnabled(False)
            if self.start_dropdown_card:
                self.start_dropdown_card.configItem.clear()
            if self.end_dropdown_card:
                self.end_dropdown_card.configItem.clear()
                self.start_dropdown_card.setTranscript(None)
                self.end_dropdown_card.setTranscript(None)
        else:
            pass  # Other widgets don't need special clearing

    def load_data(self):
        """Load data into the widget"""
        if self.engine_type == EngineType.F5:
            self.generate_button.setEnabled(True)
            if self.transcribe_button:
                self.transcribe_button.setEnabled(False)
            self.text_input.setPlaceholderText(f"Transcript: {self.transcribe_state['transcript']} \n\nPlease Enter Your Text Now")
            logger.debug(f"{self.transcribe_state['words_info']}")

            if self.start_dropdown_card and self.end_dropdown_card:
                self.start_dropdown_card.setTranscript(self.transcribe_state['words_info'])
                self.end_dropdown_card.setTranscript(self.transcribe_state['words_info'])

                for word_info in self.transcribe_state['words_info']:
                    self.start_dropdown_card.configItem.addItem(f"{word_info['word']}\t{word_info['start']}", userData=word_info)
                    self.end_dropdown_card.configItem.addItem(f"{word_info['word']}\t{word_info['end']}", userData=word_info)

                self.end_dropdown_card.configItem.setCurrentIndex(self.end_dropdown_card.configItem.count() - 1)
        else:
            # For other widgets, just update the placeholder text
            if isinstance(self.transcribe_state, dict) and 'transcript' in self.transcribe_state:
                self.transcribe_state = self.transcribe_state['transcript']
            self.text_input.setPlaceholderText(f'Transcript: {self.transcribe_state} \n\nPlease Enter Your Text Now')
            self.generate_button.setEnabled(True)
