from qfluentwidgets import (
    FluentIcon as FIF, RangeSettingCard
)

from src.config.config import cfg
from src.settings.generic_settings import GenericSettings
from ui.cards import RangeSettingCardScaled


class SparkSettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.temperature_card = RangeSettingCardScaled(
            cfg.spark_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Control randomness in generation'),
            parent=self.settings_group
        )

        self.top_p_card = RangeSettingCardScaled(
            cfg.spark_top_p,
            FIF.UP,
            self.tr('Top P'),
            self.tr('Higher values give more creativity in generation'),
            parent=self.settings_group
        )

        self.top_k_card = RangeSettingCard(
            cfg.spark_top_k,
            FIF.UP,
            self.tr('Top K'),
            self.tr('Lower values make it more predictable and coherent'),
            parent=self
        )

        self.max_new_tokens_card = RangeSettingCard(
            cfg.spark_max_new_tokens,
            FIF.UP,
            self.tr('Max New Tokens'),
            self.tr('The max number of new speech tokens, settings too low can cause issues'),
            parent=self
        )

        self.__initWidget()

    def __initWidget(self):
        # add cards to group
        self.settings_group.addSettingCard(self.temperature_card)
        self.settings_group.addSettingCard(self.top_p_card)
        self.settings_group.addSettingCard(self.top_k_card)
        self.settings_group.addSettingCard(self.max_new_tokens_card)

        self.setupLayout()