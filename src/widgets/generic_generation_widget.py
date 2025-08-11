from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

from PySide6.QtWidgets import QGroupBox, QHBoxLayout
from qfluentwidgets import FluentIcon as FIF, PrimaryPushButton
from src.ui.cards import RadioSettingCard, ComboBoxWordsCard

from src.audio.audio_player import StandardAudioPlayerBar
from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.utils.logging_utils import logger
from src.widgets.generation_widget import GenerationWidget
from src.widgets.engine_widgets_config import SETTINGS_WIDGETS, HELP_WIDGETS

class GenericGenerationWidget(GenerationWidget):
    """
    A generic widget that combines functionality from all generation widgets.
    It uses the engine type to determine its behavior and appearance.
    """

    def __init__(self, parent: FallTalkApp, is_rvc: bool = False):
        super().__init__(parent=parent, text="RVC" if is_rvc else "TTS Engines")
        self.is_rvc = is_rvc
        self.engine_type = None  # Will be set by change_engine
        self.text_input.setPlaceholderText("If no reference is selected, a default will be used. Selecting a reference audio can help change the emotion of the generated speech. ")
        self.transcribe_state = None
        self.words_data = None

        # F5-specific UI elements
        self.start_dropdown_card = None
        self.end_dropdown_card = None
        self.start_and_end = None
        self.temp_and_rep = None
        self.transcribe_button = None

        self.addGenSettings()

        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVolume(100)
        self.buttons_layout = QHBoxLayout()

        # Add transcribe button for F5 in edit mode (initially hidden)
        self.transcribe_button = PrimaryPushButton(text="Transcribe Reference Audio")
        self.transcribe_button.setIcon(FIF.PENCIL_INK)
        self.transcribe_button.clicked.connect(self.transcribe)
        self.transcribe_button.setEnabled(False)
        self.transcribe_button.setVisible(False)
        self.buttons_layout.addWidget(self.transcribe_button, stretch=1)

        # Add generate button
        self.generate_button = PrimaryPushButton(text="Generate Audio")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.generate_audio)

        self.buttons_layout.addWidget(self.generate_button, stretch=1)

        # Add settings and help widgets (will be dynamically changed)
        self.settings_widget = None
        self.help_widget = None

        self.buttons_layout.addWidget(self.settings_button)
        self.buttons_layout.addWidget(self.help_button)

        self.boxLayout.addLayout(self.buttons_layout)
        self.addToFrame(self.media_player)

        self.setEnabled(False)

    def change_engine(self, engine_type: EngineType):
        """Change the engine and update UI accordingly."""
        self.engine_type = engine_type
        
        # Clear previous widgets
        if self.settings_widget:
            self.settings_widget.setParent(None)
            self.settings_widget.deleteLater()
            self.settings_widget = None
            
        if self.help_widget:
            self.help_widget.setParent(None)
            self.help_widget.deleteLater()
            self.help_widget = None
            
        # Clear F5-specific UI if exists
        if self.start_and_end:
            self.start_and_end.setParent(None)
            self.start_and_end.deleteLater()
            self.start_and_end = None
            
        if self.temp_and_rep:
            self.temp_and_rep.setParent(None)
            self.temp_and_rep.deleteLater()
            self.temp_and_rep = None
            
        if self.transcribe_button:
            self.transcribe_button.setVisible(False)
            self.transcribe_button.setEnabled(False)

        # Setup F5-specific UI if needed
        if engine_type == EngineType.F5:
            self.setup_f5_specific_ui()
            # Show F5-specific elements based on mode
            is_edit_mode = cfg.get(cfg.f5_mode) == "edit"
            if self.start_and_end:
                self.start_and_end.setVisible(is_edit_mode)
            if self.temp_and_rep:
                self.temp_and_rep.setVisible(is_edit_mode)
            if self.transcribe_button:
                self.transcribe_button.setVisible(is_edit_mode)
        else:
            # Hide F5-specific elements for other engines
            if self.start_and_end:
                self.start_and_end.setVisible(False)
            if self.temp_and_rep:
                self.temp_and_rep.setVisible(False)
            if self.transcribe_button:
                self.transcribe_button.setVisible(False)

        # Add settings and help widgets based on engine type
        if engine_type in SETTINGS_WIDGETS:
            self.settings_widget = SETTINGS_WIDGETS[engine_type](self)
            self.settings_drawer.addWidget(self.settings_widget)

        if engine_type in HELP_WIDGETS:
            self.help_widget = HELP_WIDGETS[engine_type](self)
            self.help_drawer.addWidget(self.help_widget)

        # Update button connections for F5
        if engine_type == EngineType.F5:
            # Disconnect previous connections
            try:
                self.generate_button.clicked.disconnect()
            except RuntimeError:
                pass  # No connections to disconnect
            self.generate_button.clicked.connect(lambda: self.parent.generate_audio(transcribe_state=self.transcribe_state))
        else:
            # Disconnect previous connections
            try:
                self.generate_button.clicked.disconnect()
            except RuntimeError:
                pass  # No connections to disconnect
            self.generate_button.clicked.connect(self.parent.generate_audio)

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