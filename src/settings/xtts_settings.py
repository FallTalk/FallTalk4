from qfluentwidgets import (
    FluentIcon as FIF, SwitchSettingCard, RangeSettingCard
)

from src.config.config import cfg
from src.settings.generic_settings import GenericSettings
from src.ui.cards import RangeSettingCardScaled
from src.utils.icons import FallTalkIcons


class XTTSSettings(GenericSettings):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.low_vram_card = SwitchSettingCard(
            FallTalkIcons.RAM.icon(),
            self.tr('Low VRAM'),
            self.tr('Move data between CPU and GPU memory as needed'),
            configItem=cfg.low_vram,
        )

        self.deepspeed_card = SwitchSettingCard(
            FIF.SPEED_HIGH,
            self.tr('DeepSpeed Enabled'),
            self.tr('Speed Boost Library for NVIDIA cards, Recommended'),
            configItem=cfg.deepspeed_enabled,
        )

        self.temperature_card = RangeSettingCardScaled(
            cfg.model_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Randomness, 1 = balanced, 0 = disabled'),
        )

        self.repetition_penalty_card = RangeSettingCard(
            cfg.model_repetition,
            FallTalkIcons.LOOP.icon(),
            self.tr('Repetition Penalty'),
            self.tr('Discourage the model from repeating the same words or phrases multiple times in the same sounding way'),
        )



    def __initWidget(self):
        # add cards to group
        self.settings_group.addSettingCard(self.low_vram_card)
        self.settings_group.addSettingCard(self.deepspeed_card)
        self.settings_group.addSettingCard(self.temperature_card)
        self.settings_group.addSettingCard(self.repetition_penalty_card)

        self.setupLayout()
