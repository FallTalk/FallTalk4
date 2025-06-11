from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.widgets import GenerationWidget

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SettingCardGroup, isDarkTheme
)
from qfluentwidgets import ScrollArea, ExpandLayout



from src.ui.cards import TextAreaCard, TextSettingCard, TextCard
from src.utils.icons import FallTalkIcons


class FishHelp(ScrollArea):

    def __init__(self, parent: GenerationWidget):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('About'), self.scroll_widget)

        self.info = TextCard(
            text=
            """
            <ul>
            <li>Fish is a versatile text-to-speech model designed for generating natural-sounding speech.</li>
            <li>It offers good performance across a variety of speaking styles and content types.</li>
            <li>The model is optimized for efficiency while maintaining high-quality output.</li>
            </ul> 
            """,
            height=300
        )

        self.text = TextCard(
            text=
            """
            <ul>
            <li>For optimal results, provide text with proper punctuation and structure.</li>
            <li>The model handles both short phrases and longer paragraphs effectively.</li>
            <li>If you're generating dialogue, consider adding appropriate punctuation to indicate pauses and intonation changes.</li>
            <li>Experiment with different parameter settings to find the voice characteristics that best suit your needs.</li>
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

        self.settings_group.addSettingCard(self.text)

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