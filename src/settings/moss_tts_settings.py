from qfluentwidgets import (
    FluentIcon as FIF,
)

from src.config.config import cfg
from src.settings.generic_settings import GenericSettings
from src.ui.cards import RangeSettingCardScaled


class MossTTSSettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.temperature_card = RangeSettingCardScaled(
            cfg.moss_temperature,
            FIF.SPEED_OFF,
            self.tr('Temperature'),
            self.tr('Higher values increase variety'),
            parent=self.settings_group,
        )

        self.top_p_card = RangeSettingCardScaled(
            cfg.moss_top_p,
            FIF.SPEED_OFF,
            self.tr('Top P'),
            self.tr('Nucleus sampling threshold'),
            parent=self.settings_group,
        )

        self.top_k_card = RangeSettingCardScaled(
            cfg.moss_top_k,
            FIF.SPEED_OFF,
            self.tr('Top K'),
            self.tr('Limit sampling to the top k choices'),
            parent=self.settings_group,
            scale=1,
        )

        self.max_new_tokens_card = RangeSettingCardScaled(
            cfg.moss_max_new_tokens,
            FIF.SPEED_OFF,
            self.tr('Max New Tokens'),
            self.tr('Upper bound on the generated audio length'),
            parent=self.settings_group,
            scale=1,
        )

        self.__initWidget()

    def __initWidget(self):
        self.settings_group.addSettingCard(self.temperature_card)
        self.settings_group.addSettingCard(self.top_p_card)
        self.settings_group.addSettingCard(self.top_k_card)
        self.settings_group.addSettingCard(self.max_new_tokens_card)
        self.setupLayout()
