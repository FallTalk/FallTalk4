from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SettingCardGroup, RangeSettingCard, isDarkTheme
)
from qfluentwidgets import ScrollArea, ExpandLayout

from src.config.config import cfg
from src.ui.cards import RangeSettingCardScaled


class OrpheusSettings(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr(''), self.scroll_widget)

        self.temperature_card = RangeSettingCardScaled(
            cfg.orpehus_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Control randomness in generation'),
            parent=self.settings_group
        )

        self.seed_card = RangeSettingCard(
            cfg.orpehus_seed,
            FIF.SETTING,
            self.tr('Seed'),
            self.tr('Random seed for generation (-1 for random)'),
            parent=self.settings_group
        )

        self.top_p_card = RangeSettingCardScaled(
            cfg.orpehus_top_p,
            FIF.UP,
            self.tr('Top P'),
            self.tr('Higher values give more creativity in generation'),
            parent=self.settings_group
        )

        self.repetition_card = RangeSettingCard(
            cfg.orpehus_repetition,
            FIF.SETTING,
            self.tr('Repetition'),
            self.tr('Control repetition in generation'),
            parent=self.settings_group
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
        self.settings_group.addSettingCard(self.temperature_card)
        self.settings_group.addSettingCard(self.seed_card)
        self.settings_group.addSettingCard(self.top_p_card)
        self.settings_group.addSettingCard(self.repetition_card)

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