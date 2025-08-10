from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SubtitleLabel, ComboBoxSettingCard, RangeSettingCard, SettingCardGroup, isDarkTheme
)
from qfluentwidgets import ScrollArea, ExpandLayout

from src.settings.generic_settings import GenericSettings
from src.config.config import cfg
from ui.cards import RangeSettingCardScaled


class CSMSettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.temperature_card = RangeSettingCardScaled(
            cfg.spark_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Control randomness in generation'),
            parent=self.settings_group
        )

        self.__initWidget()

    def __initWidget(self):
        # add cards to group
        self.settings_group.addSettingCard(self.temperature_card)

        self.setupLayout()