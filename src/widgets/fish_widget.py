from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

from PySide6.QtWidgets import QGroupBox, QHBoxLayout
from qfluentwidgets import FluentIcon as FIF, RangeSettingCard, PrimaryPushButton, ToolButton


from audio.audio_player import StandardAudioPlayerBar
from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.settings.fish_speech_settings import FishSpeechSettings
from src.ui.cards import RangeSettingCardScaled, RadioSettingCard
from src.widgets.generation_widget import GenerationWidget
from src.help.fish_help import FishHelp

class FishWidget(GenerationWidget):

    def __init__(self, parent: FallTalkApp):
        super().__init__(parent=parent, text="FishSpeech")
        self.text_input.setPlaceholderText("Please Select the 'Transcribe Reference Audio' button below")
        self.transcribe_state = None
        self.words_data = None

        self.mode_card = RadioSettingCard(
            cfg.slice_mode,
            FIF.CUT,
            self.tr('Slice Mode'),
            self.tr('How to slice the sentence for longer TTS generation'),
            texts=["No Slice", "Basic punctuation . ! ? ...", "Every punctuation", "Every 4 sentences", "Every 2 sentences"],
            parent=self
        )

        self.temperature_card = RangeSettingCardScaled(
            cfg.temperature_gpt_sovits,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Controls the randomness of the generation'),
            parent=self

        )

        self.speed_card = RangeSettingCardScaled(
            cfg.speed_gpt_sovits,
            FIF.SPEED_OFF,
            self.tr('Speed'),
            self.tr('Increase or decrease the generated audio speed'),
            parent=self

        )

        self.temp_and_speed = QGroupBox()
        self.temp_and_speed.setStyleSheet("border: none")
        self.temp_and_speed_layout = QHBoxLayout()
        self.temp_and_speed_layout.setContentsMargins(0, 0, 0, 0)
        self.temp_and_speed_layout.addWidget(self.temperature_card, 3)
        self.temp_and_speed_layout.addWidget(self.speed_card, 3)
        self.temp_and_speed.setLayout(self.temp_and_speed_layout)

        self.top_p_card = RangeSettingCardScaled(
            cfg.top_p_gpt_sovits,
            FIF.UP,
            self.tr('Top P'),
            self.tr('Higher values give more creativity in generation.'),
            parent=self
        )

        self.top_k_card = RangeSettingCard(
            cfg.top_k_gpt_sovits,
            FIF.UP,
            self.tr('Top K'),
            self.tr('Lower values make it more predictable and coherent'),
            parent=self
        )

        self.addToFrame(self.mode_card)
        self.addToFrame(self.temp_and_speed)

        self.p_and_k = QGroupBox()
        self.p_and_k.setStyleSheet("border: none")
        self.p_and_k_layout = QHBoxLayout()
        self.p_and_k_layout.setContentsMargins(0, 0, 0, 0)
        self.p_and_k_layout.addWidget(self.top_p_card, 3)
        self.p_and_k_layout.addWidget(self.top_k_card, 3)
        self.p_and_k.setLayout(self.p_and_k_layout)
        self.addToFrame(self.p_and_k)

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

        self.settings_drawer.addWidget(FishSpeechSettings(self))
        self.help_drawer.addWidget(FishHelp(self))
        self.buttons_layout.addWidget(self.settings_button)
        self.buttons_layout.addWidget(self.help_button)

        self.boxLayout.addLayout(self.buttons_layout)
        self.addToFrame(self.media_player)

        self.setVisible(cfg.engine.value == EngineType.FISH_SPEECH.value)
        self.media_player.setVisible(cfg.engine.value == EngineType.FISH_SPEECH.value)
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