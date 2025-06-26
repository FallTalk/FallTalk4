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


class OrpheusHelp(ScrollArea):

    def __init__(self, parent: GenerationWidget):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('About'), self.scroll_widget)

        self.info = TextCard(
            text=
            """
            <ul>
            <li>Orpheus is a sophisticated text-to-speech model named after the legendary musician of Greek mythology.</li>
            <li>It excels at producing highly musical and expressive speech with natural prosody.</li>
            <li>The model is particularly effective for generating speech with emotional depth and nuanced intonation.</li>
            </ul> 
            """,
            height=300
        )

        self.text = TextCard(
            text=
            """
            <ul>
            <li>For optimal results, provide text with appropriate punctuation to guide the model's pacing and intonation.</li>
            <li>The model performs exceptionally well with dialogue and narrative content that requires emotional expression.</li>
            <li>Consider using commas and other punctuation marks strategically to create natural pauses in the generated speech.</li>
            <li>Experiment with different temperature settings: higher values create more varied and expressive speech, while lower values produce more consistent results.</li>
            <li>For longer texts, breaking them into meaningful segments can help maintain consistent quality throughout the generation.</li>
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