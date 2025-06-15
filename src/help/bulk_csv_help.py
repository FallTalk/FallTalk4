from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.widgets import BaseBulkWidget

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SettingCardGroup, isDarkTheme
)
from qfluentwidgets import ScrollArea, ExpandLayout



from src.ui.cards import TextAreaCard, TextSettingCard, TextCard
from src.utils.icons import FallTalkIcons


class BulkCSVHelp(ScrollArea):

    def __init__(self, parent: 'BaseBulkWidget'):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('About'), self.scroll_widget)

        self.info = TextCard(
            text=
            """
            <ul>
            <li>The CSV Bulk Generation tool allows you to generate multiple audio files from a CSV file.</li>
            <li>This is particularly useful for creating voice lines for multiple characters or dialogue entries at once.</li>
            <li>The tool uses the currently loaded TTS engine to generate the audio files.</li>
            </ul> 
            """,
            height=200
        )

        self.text = TextCard(
            text=
            """
            <ul>
            <li>Your CSV file should have the following columns in order: filename, character, text, reference, output_dir</li>
            <li>filename: The name of the output file (optional, will be randomly generated if blank)</li>
            <li>character: The Fallout 4 game name of the character (required)</li>
            <li>text: The text you want to generate (required for TTS engines)</li>
            <li>reference: The reference FUZ file name (required for TTS engines)</li>
            <li>output_dir: The directory to save the output files (optional)</li>
            <li>You can edit entries directly in the table after loading the CSV file.</li>
            <li>The "Create FUZ" option will generate game-ready files with lip synchronization.</li>
            <li>Using RVC upscaler is recommended for enhancing the quality of the generated audio.</li>
            <li>The "Keep Only FUZ" option will delete intermediate files after generation, saving disk space.</li>
            </ul>
            """,
            height=400
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