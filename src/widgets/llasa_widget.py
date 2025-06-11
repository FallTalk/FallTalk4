from __future__ import annotations
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp


from PySide6.QtWidgets import QHBoxLayout
from qfluentwidgets import FluentIcon as FIF, PrimaryPushButton, PushButton, ToolButton


from audio.audio_player import StandardAudioPlayerBar
from settings.llasa_settings import LLASASettings
from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.widgets import GenerationWidget
from src.help.llasa_help import LlasaHelp


class LlasaWidget(GenerationWidget):

    def __init__(self, parent: FallTalkApp):
        super().__init__(parent=parent, text=EngineType.LLASA.value)
        self.text_input.setPlaceholderText("Please Select the 'Transcribe Reference Audio' button below")
        self.transcribe_state = None
        self.words_data = None

        self.addGenSettings()

        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVolume(100)
        self.buttons_layout = QHBoxLayout()
        self.transcribe_button = PrimaryPushButton(text="Transcribe Reference Audio")
        self.transcribe_button.setIcon(FIF.PENCIL_INK)
        self.transcribe_button.clicked.connect(self.transcribe)
        self.transcribe_button.setVisible(False)
        self.buttons_layout.addWidget(self.transcribe_button, stretch=1)
        self.generate_button = PrimaryPushButton(text="Generate Audio")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.generate_audio)
        self.buttons_layout.addWidget(self.generate_button, stretch=1)

        self.settings_drawer.addWidget(LLASASettings(self))
        self.help_drawer.addWidget(LlasaHelp(self))

        self.buttons_layout.addWidget(self.settings_button)
        self.buttons_layout.addWidget(self.help_button)

        self.buttons_layout.addWidget(self.settings_button)

        self.boxLayout.addLayout(self.buttons_layout)
        self.addToFrame(self.media_player)

        self.setVisible(cfg.engine.value == EngineType.LLASA.value)
        self.media_player.setVisible(cfg.engine.value == EngineType.LLASA.value)
        self.setEnabled(False)

    def onReferenceSelect(self):
        if self.parent.tts_engine and self.parent.tts_engine.is_base:
            self.generate_button.setEnabled(False)
            self.transcribe_button.setEnabled(True)
        elif self.parent.tts_engine:
            self.generate_button.setEnabled(True)
            self.transcribe_button.setVisible(False)

    def transcribe(self):
        self.parent.transcribe(self)

    def clear(self):
        self.generate_button.setEnabled(False)
        self.transcribe_button.setEnabled(False)

    def load_data(self):
        self.transcribe_state = self.transcribe_state['transcript']
        self.text_input.setPlaceholderText(f'Transcript: {self.transcribe_state} \n\nPlease Enter Your Text Now')
        self.generate_button.setEnabled(True)
        self.transcribe_button.setEnabled(False)