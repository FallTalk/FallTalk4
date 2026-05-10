from qfluentwidgets import (
    FluentIcon as FIF,
)

from src.config.config import cfg
from src.settings.generic_settings import GenericSettings
from src.ui.cards import RangeSettingCardScaled, TextSettingCard


class OmniVoiceSettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.num_step_card = RangeSettingCardScaled(
            cfg.omnivoice_num_step,
            FIF.SPEED_OFF,
            self.tr('Diffusion Steps'),
            self.tr('More steps can improve quality at the cost of speed'),
            parent=self.settings_group,
            scale=1,
        )

        self.speed_card = RangeSettingCardScaled(
            cfg.omnivoice_speed,
            FIF.SPEED_OFF,
            self.tr('Speed'),
            self.tr('Speaking rate, 100 = normal speed'),
            parent=self.settings_group,
            scale=100,
        )

        self.instruct_card = TextSettingCard(
            cfg.omnivoice_instruct,
            FIF.CHAT,
            self.tr('Voice Design'),
            self.tr('Optional voice description, e.g. "female, British accent"'),
            placeholder="Type a voice description",
            parent=self.settings_group
        )

        self.__initWidget()

    def __initWidget(self):
        self.settings_group.addSettingCard(self.num_step_card)
        self.settings_group.addSettingCard(self.speed_card)
        self.settings_group.addSettingCard(self.instruct_card)
        self.setupLayout()
