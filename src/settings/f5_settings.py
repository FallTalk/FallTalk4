from qfluentwidgets import (
    FluentIcon as FIF
)

from src.config.config import cfg
from src.settings.generic_settings import GenericSettings
from src.ui.cards import RangeSettingCardScaled


class F5Settings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.speed_card = RangeSettingCardScaled(
            cfg.f5_speed,
            FIF.SPEED_OFF,
            self.tr('Speed Factor'),
            self.tr('Adjust the speed of generated audio'),
            parent=self.settings_group
        )

        self.__initWidget()

    def __initWidget(self):
        # add cards to group
        # self.settings_group.addSettingCard(self.mode_card)
        self.settings_group.addSettingCard(self.speed_card)

        self.setupLayout()

