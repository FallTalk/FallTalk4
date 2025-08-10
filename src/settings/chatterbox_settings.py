from qfluentwidgets import (
    FluentIcon as FIF
)

from src.config.config import cfg
from src.settings.generic_settings import GenericSettings
from ui.cards import RangeSettingCardScaled


class ChatterboxSettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.temperature_card = RangeSettingCardScaled(
            cfg.chatterbox_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Control randomness in generation'),
            parent=self.settings_group
        )

        self.top_p_card = RangeSettingCardScaled(
            cfg.chatterbox_top_p,
            FIF.UP,
            self.tr('Top P'),
            self.tr('Higher values give more creativity in generation'),
            parent=self.settings_group
        )

        self.top_k_card = RangeSettingCardScaled(
            cfg.chatterbox_min_p,
            FIF.DOWN,
            self.tr('Min P'),
            self.tr('Lower values make it more predictable and coherent'),
            parent=self,
            scale=1000.0
        )

        self.max_new_tokens_card = RangeSettingCardScaled(
            cfg.chatterbox_exaggeration,
            FIF.EXPRESSIVE_INPUT_ENTRY,
            self.tr('Exaggeration'),
            self.tr('How expressive and exaggerated should the generation be. Higher exaggeration tends to speed up speech'),
            parent=self
        )

        self.cfg_weight_card = RangeSettingCardScaled(
            cfg.chatterbox_cfg_weight,
            FIF.STOP_WATCH,
            self.tr('Config Weight'),
            self.tr('Lower values slow down generation speed'),
            parent=self
        )

        self.repetition_penalty_card = RangeSettingCardScaled(
            cfg.chatterbox_repetition_penalty,
            FIF.MORE,
            self.tr('Repetition penalty'),
            self.tr('How diverse should the generation be'),
            scale=10.0
        )

        self.__initWidget()


    def __initWidget(self):
        # add cards to group
        self.settings_group.addSettingCard(self.temperature_card)
        self.settings_group.addSettingCard(self.top_p_card)
        self.settings_group.addSettingCard(self.top_k_card)
        self.settings_group.addSettingCard(self.max_new_tokens_card)
        self.settings_group.addSettingCard(self.repetition_penalty_card)
        self.settings_group.addSettingCard(self.cfg_weight_card)

        self.setupLayout()