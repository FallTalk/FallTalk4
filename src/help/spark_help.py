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


class SparkHelp(ScrollArea):

    def __init__(self, parent: GenerationWidget):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('About'), self.scroll_widget)

        self.info = TextCard(
            text=
            """
            <ul>
            <li>Spark is a powerful text-to-speech model designed for generating highly natural and expressive speech.</li>
            <li>It uses advanced neural network architectures to produce speech with realistic intonation and rhythm.</li>
            <li>The model is capable of handling a wide range of speaking styles and content types.</li>
            </ul> 
            """,
            height=300
        )

        self.text = TextCard(
            text=
            """
            <ul>
            <li>For best results, provide well-structured text with appropriate punctuation.</li>
            <li>The model performs particularly well with conversational content and narrative text.</li>
            <li>When generating longer passages, consider breaking them into logical segments for more consistent quality.</li>
            <li>Experiment with different temperature settings to find the right balance between consistency and expressiveness.</li>
            <li>Using RVC upscaler can significantly enhance the clarity and naturalness of the generated speech.</li>
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