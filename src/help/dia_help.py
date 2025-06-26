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


class DIAHelp(ScrollArea):

    def __init__(self, parent: GenerationWidget):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('About'), self.scroll_widget)

        self.info = TextCard(
            text=
            """
            <ul>
            <li>DIA (Dynamic Intonation Adjustment) is a text-to-speech model that focuses on natural intonation patterns.</li>
            <li>It provides high-quality voice synthesis with dynamic pitch and rhythm control.</li>
            <li>The model is particularly good at capturing emotional nuances in speech.</li>
            </ul> 
            """,
            height=300
        )

        self.text = TextCard(
            text=
            """
            <ul>
            <li>Temperature: Controls randomness in generation. Higher values (e.g., 1.0) create more varied speech, while lower values (e.g., 0.5) produce more consistent results.</li>
            <li>Speed: Adjusts the pace of the generated speech. Increase for faster speech, decrease for slower delivery.</li>
            <li>Top P and Top K: Fine-tune the creativity and coherence of the generated speech. Experiment with these values to find the right balance.</li>
            <li>Slice Mode: For longer texts, choose how to break down the input for processing. "Basic punctuation" works well for most cases.</li>
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