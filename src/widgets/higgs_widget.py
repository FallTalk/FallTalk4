from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

from PySide6.QtWidgets import QHBoxLayout
from qfluentwidgets import FluentIcon as FIF, PrimaryPushButton

from audio.audio_player import StandardAudioPlayerBar
from src.config.config import cfg
from src.utils.logging_utils import logger
from src.widgets.generation_widget import GenerationWidget
from src.enums.engine_type import EngineType
from src.settings.higgs_settings import HiggsSettings
from src.help.higgs_help import HiggsHelp

class HiggsWidget(GenerationWidget):

    def __init__(self, parent: FallTalkApp):
        super().__init__(parent=parent, text="Higgs")
        self.text_input.setPlaceholderText("If no reference is selected, a default will be used. Selecting a reference audio can help change the voice characteristics of the generated speech.")
        self.transcribe_state = None
        self.words_data = None

        self.addGenSettings()

        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVolume(100)
        self.buttons_layout = QHBoxLayout()

        self.generate_button = PrimaryPushButton(text="Generate Audio")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(lambda: self.parent.generate_audio(transcribe_state=self.transcribe_state))
        self.buttons_layout.addWidget(self.generate_button, stretch=1)

        self.settings_drawer.addWidget(HiggsSettings(self))
        self.help_drawer.addWidget(HiggsHelp(self))
        self.buttons_layout.addWidget(self.settings_button)
        self.buttons_layout.addWidget(self.help_button)

        self.boxLayout.addLayout(self.buttons_layout)
        self.addToFrame(self.media_player)

        self.setVisible(cfg.engine.value == EngineType.HIGGS.value)
        self.media_player.setVisible(cfg.engine.value == EngineType.HIGGS.value)
        self.setEnabled(True)

    def transcribe(self):
        self.parent.transcribe(self)

    def onReferenceSelect(self):
        self.generate_button.setEnabled(True)

    def clear(self):
        pass

    def load_data(self):
        self.generate_button.setEnabled(True)
        self.text_input.setPlaceholderText(f"Transcript: {self.transcribe_state['transcript']} \n\nPlease Enter Your Text Now")
        logger.debug(f"{self.transcribe_state}")