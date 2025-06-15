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


class BulkRVCHelp(ScrollArea):

    def __init__(self, parent: 'BaseBulkWidget'):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('About'), self.scroll_widget)

        self.info = TextCard(
            text=
            """
            <ul>
            <li>The RVC Bulk Generation tool allows you to apply RVC (Retrieval-based Voice Conversion) to a folder of audio files.</li>
            <li>This is particularly useful for creating character voice replacements for Fallout 4 mods.</li>
            <li>For example, you can extract all voice lines for a character and convert them to sound like another character.</li>
            </ul> 
            """,
            height=200
        )

        self.text = TextCard(
            text=
            """
            <ul>
            <li>Select a folder containing WAV or XWM files that you want to process.</li>
            <li>Choose a character voice from the dropdown menu to apply to the audio files.</li>
            <li>The "Include Sub Directories" option allows processing nested folders in a single operation.</li>
            <li>The "Replace" option will overwrite original files - use with caution.</li>
            <li>The "Create FUZ" option will generate game-ready files with lip synchronization.</li>
            <li>The "Keep Only FUZ" option will delete intermediate files after generation, saving disk space.</li>
            <li>The "Use Existing LIP" option will use existing LIP files if available, or generate new ones if not.</li>
            <li>You can adjust the number of processing threads to optimize performance based on your system capabilities.</li>
            <li>Advanced RVC settings are available in the settings panel (gear icon).</li>
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