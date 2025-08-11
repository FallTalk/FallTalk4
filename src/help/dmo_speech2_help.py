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


class DMSpeech2Help(ScrollArea):

    def __init__(self, parent: GenerationWidget):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('About'), self.scroll_widget)

        self.info = TextCard(
            text=
            """
            General Use:
                DMSpeech2 is a high-quality text-to-speech engine that can generate natural-sounding speech.
                It uses a reference audio to capture the voice characteristics and speaking style of the speaker.
                The default settings work well for most prompts.
        
            Tips for Best Results:
                Provide clear reference audio with minimal background noise.
                Use well-punctuated text for more natural speech patterns.
                Shorter prompts generally produce more consistent results.
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
            <li>Using RVC upscaler can significantly enhance the clarity and naturalness of the generated speech.</li>
            <li>Reference audio quality significantly impacts the output quality - use clear, clean recordings.</li>
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