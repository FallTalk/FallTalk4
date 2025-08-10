from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SettingCardGroup, RangeSettingCard, isDarkTheme, SwitchSettingCard
)
from qfluentwidgets import ScrollArea, ExpandLayout

from src.settings.generic_settings import GenericSettings
from src.config.config import cfg
from src.ui.cards import RangeSettingCardScaled, RadioSettingCard
from src.utils.icons import FallTalkIcons


class DIASettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.temperature_card = RangeSettingCardScaled(
            cfg.dia_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Control randomness in generation'),
            parent=self.settings_group
        )

        self.top_p_card = RangeSettingCardScaled(
            cfg.dia_top_p,
            FIF.UP,
            self.tr('Top P'),
            self.tr('Higher values give more creativity in generation'),
            parent=self.settings_group
        )

        self.top_k_card = RangeSettingCard(
            cfg.dia_top_k,
            FIF.UP,
            self.tr('Top K'),
            self.tr('Lower values make it more predictable and coherent'),
            parent=self
        )

        self.__initWidget()

    def __initWidget(self):
        # add cards to group
        self.settings_group.addSettingCard(self.temperature_card)
        self.settings_group.addSettingCard(self.top_p_card)
        self.settings_group.addSettingCard(self.top_k_card)

        self.setupLayout()