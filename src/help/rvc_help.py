from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.widgets import BaseRVCWidget

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    SettingCardGroup, isDarkTheme
)
from qfluentwidgets import ScrollArea, ExpandLayout



from src.ui.cards import TextCard


class RVCHelp(ScrollArea):

    def __init__(self, parent: 'BaseRVCWidget'):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('About'), self.scroll_widget)
        self.modes_group = SettingCardGroup(self.tr('RVC Modes'), self.scroll_widget)

        self.info = TextCard(
            text=
            """
            <ul>
            <li>RVC (Retrieval-based Voice Conversion) is a powerful tool for transforming audio from one voice to another.</li>
            <li>It allows you to convert any voice input into the voice of a character from Fallout 4 or other custom voices.</li>
            <li>This is particularly useful for creating custom voice lines for mods or for voice acting with character voices.</li>
            </ul> 
            """,
            height=200
        )

        self.modes = TextCard(
            text=
            """
            <ul>
            <li><strong>Microphone:</strong> Record your voice directly and convert it to a character voice in real-time.</li>
            <li><strong>File:</strong> Convert an existing audio file (WAV or MP3) to a character voice.</li>
            <li><strong>Edge TTS:</strong> Use Microsoft's free text-to-speech service as a base voice, then convert it to a character voice.</li>
            <li><strong>ElevenLabs:</strong> Use ElevenLabs' high-quality text-to-speech service as a base voice, then convert it to a character voice (requires API key).</li>
            </ul>
            """,
            height=250
        )

        self.text = TextCard(
            text=
            """
            <ul>
            <li>For best results, ensure your input audio is clear and has minimal background noise.</li>
            <li>Adjust the pitch setting if the voice sounds unnatural - this is especially useful when converting between genders.</li>
            <li>The "Index Influence Ratio" controls how much the voice characteristics are applied. Higher values provide more detail but may introduce artifacts.</li>
            <li>The "Filter Radius" can help reduce breathing sounds in the output. Values of 3 or higher are recommended for this purpose.</li>
            <li>The "Autotune" option can improve singing voice conversions by correcting pitch.</li>
            <li>For longer audio files, enable "Split Audio" to process the file in chunks for better results.</li>
            <li>The "Create FUZ" option will generate game-ready files with lip synchronization.</li>
            <li>Advanced RVC settings are available in the settings panel (gear icon).</li>
            </ul>
            """,
            height=350
        )

        self.__initWidget()


    def __initWidget(self):
        self.resize(1000, 800)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 0, 0, 20)
        self.setWidget(self.scroll_widget)
        self.setWidgetResizable(True)

        # initialize style sheet
        self.__setQss()

        # initialize layout
        self.__initLayout()
        self.__connectSignalToSlot()

    def __initLayout(self):
        # add cards to group
        self.settings_group.addSettingCard(self.info)
        self.modes_group.addSettingCard(self.modes)
        self.settings_group.addSettingCard(self.text)

        # add setting card group to layout
        self.expand_layout.setSpacing(28)
        self.expand_layout.setContentsMargins(15, 0, 15, 0)
        self.expand_layout.addWidget(self.settings_group)
        self.expand_layout.addWidget(self.modes_group)

    def __setQss(self):
        """ set style sheet """
        self.scroll_widget.setObjectName('scrollWidget')

        theme = 'dark' if isDarkTheme() else 'light'
        with open(f'resource/qss/{theme}/setting_interface.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def __connectSignalToSlot(self):
        pass