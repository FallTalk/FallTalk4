from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.widgets.generation_widget import GenerationWidget

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    SettingCardGroup, isDarkTheme
)
from qfluentwidgets import ScrollArea, ExpandLayout

from src.ui.cards import TextCard


class OmniVoiceHelp(ScrollArea):

    def __init__(self, parent: GenerationWidget):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('About OmniVoice'), self.scroll_widget)

        self.info = TextCard(
            text="""
            <ul>
            <li>OmniVoice is a massively multilingual zero-shot TTS model with voice cloning and voice design.</li>
            <li>It supports reference audio cloning and can also generate from a descriptive voice prompt.</li>
            <li>Use short reference clips for best results. Long prompts can slow inference and reduce cloning quality.</li>
            </ul>
            """,
            height=300
        )

        self.tips = TextCard(
            text="""
            <ul>
            <li>Provide 3 to 10 seconds of clean reference audio when using voice cloning.</li>
            <li>If you do not provide reference text, OmniVoice will auto-transcribe the clip.</li>
            <li>Use the Voice Design field when you want a generated voice instead of cloning a specific speaker.</li>
            <li>The speed and diffusion step controls trade off latency against consistency.</li>
            </ul>
            """,
            height=300
        )

        self.__initWidget()

    def __initWidget(self):
        self.resize(1000, 800)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 0, 0, 20)
        self.setWidget(self.scroll_widget)
        self.setWidgetResizable(True)

        self.__setQss()
        self.__initLayout()
        self.__connectSignalToSlot()

    def __initLayout(self):
        self.settings_group.addSettingCard(self.info)
        self.settings_group.addSettingCard(self.tips)
        self.expand_layout.setSpacing(28)
        self.expand_layout.setContentsMargins(15, 0, 15, 0)
        self.expand_layout.addWidget(self.settings_group)

    def __setQss(self):
        self.scroll_widget.setObjectName('scrollWidget')

        theme = 'dark' if isDarkTheme() else 'light'
        with open(f'resource/qss/{theme}/setting_interface.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def __connectSignalToSlot(self):
        pass
