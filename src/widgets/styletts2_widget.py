from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

from PySide6.QtWidgets import QHBoxLayout
from qfluentwidgets import FluentIcon as FIF, PushButton, PrimaryPushButton


from audio.audio_player import StandardAudioPlayerBar
from src.widgets.generation_widget import GenerationWidget
from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.settings.styletts2_settings import StyleTTS2Settings


class StyleTTS2Widget(GenerationWidget):

    def __init__(self, parent: FallTalkApp):
        super().__init__(parent=parent, text="StyleTTS2")
        self.addGenSettings()
        self.text_input.setPlaceholderText("Please enter text")

        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVolume(100)
        self.generate_button = PrimaryPushButton(text="Generate Audio")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.generate_audio)

        # Add to layout
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(self.settings_button, stretch=1)
        self.buttons_layout.addWidget(self.generate_button, stretch=5)
        self.settings_drawer.addWidget(StyleTTS2Settings(self))
        self.help_drawer.addWidget(StyleTTS2Settings(self))
        self.buttons_layout.addWidget(self.settings_button)
        self.buttons_layout.addWidget(self.help_button)
        self.boxLayout.addLayout(self.buttons_layout)
        self.addToFrame(self.media_player)
        
        self.setVisible(cfg.engine.value == EngineType.STYLE_TTS2.value)
        self.media_player.setVisible(cfg.engine.value == EngineType.STYLE_TTS2.value)
        self.setEnabled(False)
        
