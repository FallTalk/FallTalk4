from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SettingCardGroup, isDarkTheme, SwitchSettingCard
)
from qfluentwidgets import ScrollArea, ExpandLayout

from src.config.config import cfg
from ui.cards import SpinSettingCard


class GenericSettings(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('Engine Settings'), self.scroll_widget)

        # Text processing settings
        self.text_processing_group = SettingCardGroup(self.tr('Text Processing'), self.scroll_widget)

        self.max_text_size_card = SpinSettingCard(
            cfg.max_text_size,
            FIF.CARE_UP_SOLID,
            self.tr('Max Characters'),
            self.tr('Maximum number of characters to process in a single generation'),
            parent=self.text_processing_group,
            step=10
        )

        self.min_chunk_size_card = SpinSettingCard(
            cfg.min_chunk_size,
            FIF.CARE_DOWN_SOLID,
            self.tr('Min Characters'),
            self.tr('Minimum number of characters needed'),
            parent=self.text_processing_group,
            step=5        )

        self.lowercase_conversion_card = SwitchSettingCard(
            FIF.FONT_SIZE,
            self.tr('Lowercase Conversion'),
            self.tr('Convert all text to lowercase before processing'),
            configItem=cfg.lowercase_conversion,
            parent=self.text_processing_group
        )

        self.whitespace_normalization_card = SwitchSettingCard(
            FIF.QUICK_NOTE,
            self.tr('Whitespace Normalization'),
            self.tr('Strip extra spaces and newlines from text'),
            configItem=cfg.whitespace_normalization,
            parent=self.text_processing_group
        )

        self.dot_letter_fix_card = SwitchSettingCard(
            FIF.MORE,
            self.tr('Dot-Letter Fix'),
            self.tr('Convert "J.R.R." to "J R R" to improve initialisms and names'),
            configItem=cfg.dot_letter_fix,
            parent=self.text_processing_group
        )

        self.inline_reference_removal_card = SwitchSettingCard(
            FIF.REMOVE,
            self.tr('Inline Reference Removal'),
            self.tr('Remove numbers after sentence-ending punctuation (e.g., .188 or ."3)'),
            configItem=cfg.inline_reference_removal,
            parent=self.text_processing_group
        )

    def setupLayout(self):
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
        # Add cards to text processing group
        self.text_processing_group.addSettingCard(self.max_text_size_card)
        self.text_processing_group.addSettingCard(self.min_chunk_size_card)
        self.text_processing_group.addSettingCard(self.lowercase_conversion_card)
        self.text_processing_group.addSettingCard(self.whitespace_normalization_card)
        self.text_processing_group.addSettingCard(self.dot_letter_fix_card)
        self.text_processing_group.addSettingCard(self.inline_reference_removal_card)


        # add setting card group to layout
        self.expand_layout.setSpacing(28)
        self.expand_layout.setContentsMargins(15, 0, 15, 0)
        self.expand_layout.addWidget(self.settings_group)
        self.expand_layout.addWidget(self.text_processing_group)


    def __setQss(self):
        """ set style sheet """
        self.scroll_widget.setObjectName('scrollWidget')

        theme = 'dark' if isDarkTheme() else 'light'
        with open(f'resource/qss/{theme}/setting_interface.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def __connectSignalToSlot(self):
        pass