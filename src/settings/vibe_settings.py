from qfluentwidgets import (
    FluentIcon as FIF, SwitchSettingCard, RangeSettingCard
)

from src.config.config import cfg
from src.settings.generic_settings import GenericSettings
from ui.cards import RangeSettingCardScaled, RadioSettingCard


class VibeSettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.temperature_card = RangeSettingCardScaled(
            cfg.vibe_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Control randomness in generation'),
            parent=self.settings_group
        )

        self.inference_steps = RangeSettingCard(
            cfg.vibe_inference_steps,
            FIF.UP,
            self.tr('Inference Steps'),
            self.tr('Increases generation time'),
            parent=self.settings_group
        )

        self.top_p_card = RangeSettingCardScaled(
            cfg.vibe_top_p,
            FIF.UP,
            self.tr('Top P'),
            self.tr('Higher values give more creativity in generation'),
            parent=self.settings_group
        )

        self.sample = SwitchSettingCard(
            FIF.QUESTION,
            self.tr('Use Sampling'),
            self.tr('Allows more randomness but is less stable'),
            configItem=cfg.vibe_dosmaple,
        )

        self.cfg_weight_card = RangeSettingCardScaled(
            cfg.vibe_cfg_scale,
            FIF.STOP_WATCH,
            self.tr('Config Scale'),
            self.tr('CFG (Classifier-Free Guidance) scale for generation'),
            parent=self
        )

        self.mode_card = RadioSettingCard(
            cfg.vibe_mode,
            FIF.CUT,
            self.tr('Model Size'),
            self.tr('Choose the model size to use (requires restart)'),
            texts=["1.5B (7GB VRAM)", "7B (24 GB VRAM)"],
            parent=self.settings_group
        )

        self.__initWidget()


    def __initWidget(self):
        # add cards to group
        self.settings_group.addSettingCard(self.sample)
        self.settings_group.addSettingCard(self.cfg_weight_card)
        self.settings_group.addSettingCard(self.temperature_card)
        self.settings_group.addSettingCard(self.top_p_card)
        self.settings_group.addSettingCard(self.inference_steps)
        self.settings_group.addSettingCard(self.mode_card)

        # self.settings_group.addSettingCard(self.top_k_card)
        # self.settings_group.addSettingCard(self.max_new_tokens_card)
        # self.settings_group.addSettingCard(self.repetition_penalty_card)

        self.setupLayout()