from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

from PySide6.QtWidgets import QGroupBox, QHBoxLayout
from qfluentwidgets import FluentIcon as FIF, PrimaryPushButton, PushButton, ToolButton


from audio.audio_player import StandardAudioPlayerBar
from src.config.config import cfg
from src.utils.icons import FallTalkIcons
from src.utils.logging_utils import logger
from src.ui.cards import RangeSettingCardScaled, RadioSettingCard, ComboBoxWordsCard
from src.widgets.generation_widget import GenerationWidget
from src.enums.engine_type import EngineType
from src.settings.f5_settings import F5Settings
from src.help.f5_help import F5Help

class F5Widget(GenerationWidget):

    def __init__(self, parent: FallTalkApp):
        super().__init__(parent=parent, text="F5")
        self.text_input.setPlaceholderText("Please Select Reference Audio before Generation")
        self.transcribe_state = None
        self.words_data = None

        self.mode_card = RadioSettingCard(
            cfg.f5_mode,
            FIF.DEVELOPER_TOOLS,
            self.tr('Generation Mode'),
            self.tr('How should we generate text'),
            texts=["Edit", "TTS"],
        )

        self.mode_card.optionChanged.connect(self.mode_changed)
        # self.edit_mode_card = RadioSettingCard(
        #     cfg.edit_mode,
        #     FIF.SETTING,
        #     self.tr('Editing Mode'),
        #     self.tr('What to do with the selected first and last word'),
        #     texts=["Replace Half", "Replace Completely"],
        # )

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
        # self.edit_mode_card.setVisible(cfg.get(cfg.mode) == "edit")
        self.addToFrame(self.start_and_end)

        # self.addToFrame(self.edit_mode_card)

        self.temp_and_rep = QGroupBox()
        self.temp_and_rep.setStyleSheet("border: none")
        self.temp_and_rep_layout = QHBoxLayout()
        self.temp_and_rep_layout.setContentsMargins(0, 0, 0, 0)
        self.temp_and_rep.setLayout(self.temp_and_rep_layout)
        self.temp_and_rep.setVisible(cfg.get(cfg.f5_mode) == "edit")
        self.addToFrame(self.temp_and_rep)

        self.addGenSettings()

        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVolume(100)
        self.buttons_layout = QHBoxLayout()
        self.transcribe_button = PrimaryPushButton(text="Transcribe Reference Audio")
        self.transcribe_button.setIcon(FIF.PENCIL_INK)
        self.transcribe_button.clicked.connect(self.transcribe)
        self.transcribe_button.setEnabled(False)
        self.transcribe_button.setVisible(cfg.get(cfg.f5_mode) == "edit")
        self.buttons_layout.addWidget(self.transcribe_button, stretch=1)
        self.generate_button = PrimaryPushButton(text="Generate Audio")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.generate_audio)
        # self.generate_button.setEnabled(False)
        self.buttons_layout.addWidget(self.generate_button, stretch=1)

        self.settings_drawer.addWidget(F5Settings(self))
        self.help_drawer.addWidget(F5Help(self))
        self.buttons_layout.addWidget(self.settings_button)
        self.buttons_layout.addWidget(self.help_button)

        self.boxLayout.addLayout(self.buttons_layout)
        self.addToFrame(self.media_player)

        self.setVisible(cfg.engine.value == EngineType.F5.value)
        self.media_player.setVisible(cfg.engine.value == EngineType.F5.value)
        self.setEnabled(True)

    def transcribe(self):
        self.parent.transcribe(self)

    def onReferenceSelect(self):
        self.generate_button.setEnabled(True)
        self.transcribe_button.setEnabled(True)

    def mode_changed(self, change):
        self.start_and_end.setVisible(change.value == "edit")
        self.temp_and_rep.setVisible(cfg.get(cfg.f5_mode) == "edit")
        self.transcribe_button.setVisible(cfg.get(cfg.f5_mode) == "edit")


    def clear(self):
        # self.generate_button.setEnabled(False)
        self.transcribe_button.setEnabled(False)
        self.start_dropdown_card.configItem.clear()
        self.end_dropdown_card.configItem.clear()
        self.start_dropdown_card.setTranscript(None)
        self.end_dropdown_card.setTranscript(None)

    def load_data(self):
        self.generate_button.setEnabled(True)
        self.transcribe_button.setEnabled(False)
        self.text_input.setPlaceholderText(f"Transcript: {self.transcribe_state['transcript']} \n\nPlease Enter Your Text Now")
        logger.debug(f"{self.transcribe_state['words_info']}")
        self.start_dropdown_card.setTranscript(self.transcribe_state['words_info'])
        self.end_dropdown_card.setTranscript(self.transcribe_state['words_info'])

        for word_info in self.transcribe_state['words_info']:
            self.start_dropdown_card.configItem.addItem(f"{word_info['word']}\t{word_info['start']}", userData=word_info)
            self.end_dropdown_card.configItem.addItem(f"{word_info['word']}\t{word_info['end']}", userData=word_info)

        self.end_dropdown_card.configItem.setCurrentIndex(self.end_dropdown_card.configItem.count() - 1)