from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.widgets.falltalk_widget import FallTalkWidget

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    SettingCardGroup, isDarkTheme
)
from qfluentwidgets import ScrollArea, ExpandLayout



from src.ui.cards import TextCard


class UpscaleHelp(ScrollArea):

    def __init__(self, parent: 'FallTalkWidget'):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('About'), self.scroll_widget)

        self.info = TextCard(
            text=
            """
            <ul>
            <li>The Bulk Enhancement tool allows you to process multiple audio files at once.</li>
            <li>It offers three main modes: Denoise (clean up recorded speech), Isolate Vocals (separate voice from background noise), and Upscale (improve audio quality).</li>
            <li>This tool is particularly useful for enhancing game audio files or preparing voice recordings for further processing.</li>
            </ul> 
            """,
            height=300
        )

        self.text = TextCard(
            text=
            """
            <ul>
            <li>For best results with the Upscale mode, use it on audio files that are 16 kHz or below.</li>
            <li>The Denoise mode works well for cleaning up recorded speech with background noise.</li>
            <li>Isolate Vocals is powerful for extracting voice from audio with heavy background noise or music.</li>
            <li>You can choose between 44100Hz (Fallout 4 default) and 48000Hz sample rates.</li>
            <li>The "Include Sub Directories" option allows processing nested folders in a single operation.</li>
            <li>The "Replace" option will overwrite original files - use with caution.</li>
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