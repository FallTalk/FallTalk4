from qfluentwidgets import (
    FluentIcon as FIF, RangeSettingCard
)

from src.config.config import cfg
from src.settings.generic_settings import GenericSettings
from src.ui.cards import RangeSettingCardScaled, RadioSettingCard


class LlasaSettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.mode_card = RadioSettingCard(
            cfg.llasa_mode,
            FIF.CUT,
            self.tr('Model Size'),
            self.tr('Choose the model size to use'),
            texts=["3B", "1B", "8B"],
            parent=self.settings_group
        )

        self.temperature_card = RangeSettingCardScaled(
            cfg.llasa_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Control randomness in generation'),
            parent=self.settings_group
        )

        self.top_p_card = RangeSettingCardScaled(
            cfg.llasa_top_p,
            FIF.UP,
            self.tr('Top P'),
            self.tr('Higher values give more creativity in generation'),
            parent=self.settings_group
        )

        self.max_length_card = RangeSettingCard(
            cfg.llasa_max_length,
            FIF.SETTING,
            self.tr('Max Length'),
            self.tr('Maximum length of generated text'),
            parent=self.settings_group
        )

        self.__initWidget()

    def __initWidget(self):
        # add cards to group
        # self.settings_group.addSettingCard(self.mode_card)
        self.settings_group.addSettingCard(self.temperature_card)
        self.settings_group.addSettingCard(self.top_p_card)
        self.settings_group.addSettingCard(self.max_length_card)

        self.setupLayout()
