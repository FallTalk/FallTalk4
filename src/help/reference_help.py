from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.widgets.references_widget import ReferencesWidget

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SettingCardGroup, isDarkTheme
)
from qfluentwidgets import ScrollArea, ExpandLayout



from src.ui.cards import TextAreaCard


class ReferencesHelp(ScrollArea):

    def __init__(self, parent: ReferencesWidget):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('About'), self.scroll_widget)

        self.reference = TextAreaCard(
            FIF.MIX_VOLUMES,
            self.tr('What is reference audio?'),
            text="""
            <p>This reference audio is the foundation the model uses to generate your new voice, so it has a big impact on how the final result sounds. It helps the model learn the speaker's unique vocal traits — like tone, pitch, accent, and emotional expression. Some models are already trained on a speaker’s voice and don’t always need a reference, so it’s worth trying both with and without one to see what works best. A good reference makes the difference between a natural, expressive voice and something that sounds flat or artificial.

            Each Model has different reference requirements:</p>
            <ul>
                <li>GPT SoVITs: 5-10 seconds. More than 10 seconds can lead to hallucinations.</li>
                <li>F5: 3-12 seconds.</li>
                <li>StyleTTS2: 5-10 seconds.</li>
                <li>Orpheus: 3-15 seconds.</li>
                <li>FishSpeech: 5-30 seconds.</li>
                <li>Spark: 3-15 seconds.</li>
                <li>CSM: 3-15 seconds.</li>
            </ul>  
            
            These models do not need reference audio when fully trained:
            <ul>
                <li>GPT SoVITs</li>
                <li>Orpheus</li>
                <li>Spark</li>
                <li>CSM</li>
            </ul>  
            
            If you do not select a reference, we will pick some randomly from the Fallout 4 speaker database. This will only happen if the model requires it.
            """,
            height=600
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
        self.settings_group.addSettingCard(self.reference)

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