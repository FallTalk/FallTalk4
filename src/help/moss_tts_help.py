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


class MossTTSHelp(ScrollArea):

    def __init__(self, parent: GenerationWidget):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('About MOSS-TTS'), self.scroll_widget)

        self.info = TextCard(
            text="""
            <ul>
            <li>MOSS-TTS is a high-fidelity TTS family from OpenMOSS and MOSI.AI.</li>
            <li>The flagship MOSS-TTS model supports zero-shot voice cloning and long-form synthesis.</li>
            <li>The model uses a Transformers-based processor/model pair and can work with local downloads or the Hugging Face repo.</li>
            </ul>
            """,
            height=300
        )

        self.tips = TextCard(
            text="""
            <ul>
            <li>Use the reference audio path for cloning a speaker's voice.</li>
            <li>Keep the reference clip short and clean for best quality.</li>
            <li>Lower temperature and top-p values can make output more stable.</li>
            <li>The max new tokens setting controls the longest generation window.</li>
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
