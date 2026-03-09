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

class QwenHelp(ScrollArea):

    def __init__(self, parent: GenerationWidget):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('About Qwen3 TTS'), self.scroll_widget)

        self.info = TextCard(
            text=
            """
            <ul>
            <li>Qwen3-TTS is a state-of-the-art text-to-speech model supporting multiple languages and expressive generation.</li>
            <li>It features high-quality speech synthesis with support for custom voice instructions and high-fidelity voice cloning.</li>
            <li>Available in 1.7B and 0.6B sizes, with specialized versions for custom voice generation and base cloning.</li>
            </ul> 
            """,
            height=300
        )

        self.cloning_info = TextCard(
            text=
            """
            <h3>Voice Cloning</h3>
            <ul>
            <li>For voice cloning, use the <b>1.7B-Base</b> or <b>0.6B-Base</b> models.</li>
            <li>Provide a <b>Reference Audio</b> clip and its <b>Reference Text</b> transcript for best results.</li>
            <li>If you leave Reference Text empty, it may use speaker embeddings only (quality might be reduced).</li>
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

        # initialize style sheet
        self.__setQss()

        # initialize layout
        self.__initLayout()
        self.__connectSignalToSlot()

    def __initLayout(self):
        # add cards to group
        self.settings_group.addSettingCard(self.info)
        self.settings_group.addSettingCard(self.cloning_info)

        # add setting card group to layout
        self.expand_layout.setSpacing(28)
        self.expand_layout.setContentsMargins(15, 0, 15, 0)
        self.expand_layout.addWidget(self.settings_group)

    def __setQss(self):
        """ set style sheet """
        self.scroll_widget.setObjectName('scrollWidget')

        theme = 'dark' if isDarkTheme() else 'light'
        with open(f'resource/qss/{theme}/setting_interface.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def __connectSignalToSlot(self):
        pass
