from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SettingCardGroup, RangeSettingCard, isDarkTheme
)
from qfluentwidgets import ScrollArea, ExpandLayout

from src.config.config import cfg
from src.ui.cards import RangeSettingCardScaled


class HiggsSettings(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr(''), self.scroll_widget)

        self.temperature_card = RangeSettingCardScaled(
            cfg.higgs_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Controls the randomness of the generated audio'),
            parent=self.settings_group
        )

        self.top_p_card = RangeSettingCardScaled(
            cfg.higgs_top_p,
            FIF.UP,
            self.tr('Top P'),
            self.tr('Controls the diversity of the generated audio'),
            parent=self.settings_group
        )

        self.top_k_card = RangeSettingCardScaled(
            cfg.higgs_top_k,
            FIF.UP,
            self.tr('Top K'),
            self.tr('Limits the number of tokens considered for each step'),
            parent=self.settings_group
        )

        self.max_new_tokens_card = RangeSettingCardScaled(
            cfg.higgs_max_new_tokens,
            FIF.SCROLL,
            self.tr('Max New Tokens'),
            self.tr('Maximum number of tokens to generate'),
            parent=self.settings_group
        )

        self.ras_win_len_card = RangeSettingCardScaled(
            cfg.higgs_ras_win_len,
            FIF.FLAG,
            self.tr('RAS Window Length'),
            self.tr('Window length for repetition avoidance sampling'),
            parent=self.settings_group
        )

        self.ras_win_max_num_repeat_card = RangeSettingCardScaled(
            cfg.higgs_ras_win_max_num_repeat,
            FIF.RETURN,
            self.tr('RAS Max Repeats'),
            self.tr('Maximum number of repeats allowed in RAS window'),
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
        self.settings_group.addSettingCard(self.top_p_card)
        self.settings_group.addSettingCard(self.top_k_card)
        self.settings_group.addSettingCard(self.max_new_tokens_card)
        self.settings_group.addSettingCard(self.ras_win_len_card)
        self.settings_group.addSettingCard(self.ras_win_max_num_repeat_card)

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