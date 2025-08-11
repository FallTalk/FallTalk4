from qfluentwidgets import (
    FluentIcon as FIF
)

from src.config.config import cfg
from src.settings.generic_settings import GenericSettings
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