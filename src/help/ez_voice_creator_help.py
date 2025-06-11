from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.widgets import FallTalkWidget

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    SettingCardGroup, isDarkTheme
)
from qfluentwidgets import ScrollArea, ExpandLayout



from src.ui.cards import TextCard


class EzVoiceCreatorHelp(ScrollArea):

    def __init__(self, parent: 'FallTalkWidget'):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('About'), self.scroll_widget)

        self.info = TextCard(
            text=
            """
            The ESP Voice Generator is a powerful tool for creating voice lines for Fallout 4 mods.
            
            FallTalk includes a custom made xEdit script that will export the dialogue from any mod into CSV. There is one problem with the script, xEdit will not close automatically.
            
            >> You Must close xEdit manually after the script completes. <<
            
            This tool streamlines the process of creating voiced dialogue for NPCs and characters in your mods.
            """,
            height=300
        )

        self.text = TextCard(
            text=
            """
            To get started, click the "xEdit" button to export dialogue from your mod, or select an existing CSV file.
            You can filter the table entries using the search box at the top.
            Enable "Edit Mode" to modify entries directly in the table by double-clicking on cells.
            The "Create FUZ" option will generate game-ready files with lip synchronization.
            The "Keep Only FUZ" option will delete intermediate files after generation, saving disk space.
            """,
            height=200
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