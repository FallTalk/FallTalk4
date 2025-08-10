from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SettingCardGroup, RangeSettingCard, isDarkTheme, SwitchSettingCard
)
from qfluentwidgets import ScrollArea, ExpandLayout

from src.settings.generic_settings import GenericSettings
from src.config.config import cfg
from src.ui.cards import RangeSettingCardScaled


class FishSpeechSettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.torch_compile_card = SwitchSettingCard(
            FIF.SPEED_OFF,
            self.tr('Use Torch Compile'),
            self.tr('Enable torch.compile for faster inference'),
            configItem=cfg.fish_use_torch_compile,
            parent=self.settings_group
        )

        self.repetition_card = RangeSettingCard(
            cfg.fish_repetition,
            FIF.SETTING,
            self.tr('Repetition'),
            self.tr('Control repetition in generation'),
            parent=self.settings_group
        )

        self.top_p_card = RangeSettingCardScaled(
            cfg.fish_top_p,
            FIF.UP,
            self.tr('Top P'),
            self.tr('Higher values give more creativity in generation'),
            parent=self.settings_group
        )

        self.max_length_card = RangeSettingCard(
            cfg.fish_max_length,
            FIF.SETTING,
            self.tr('Max Length'),
            self.tr('Maximum length of generated text'),
            parent=self.settings_group
        )

        self.use_cache_card = SwitchSettingCard(
            FIF.SETTING,
            self.tr('Use Memory Cache'),
            self.tr('Cache model outputs for faster generation'),
            configItem=cfg.fish_use_cache,
            parent=self.settings_group
        )

        self.iterative_prompt_card = SwitchSettingCard(
            FIF.EDIT,
            self.tr('Iterative Prompt'),
            self.tr('Use iterative prompting for better results'),
            configItem=cfg.fish_iterative_prompt,
            parent=self.settings_group
        )

        self.temperature_card = RangeSettingCardScaled(
            cfg.fish_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Control randomness in generation'),
            parent=self.settings_group
        )

        self.__initWidget()

    def __initWidget(self):
        # add cards to group
        self.settings_group.addSettingCard(self.torch_compile_card)
        self.settings_group.addSettingCard(self.repetition_card)
        self.settings_group.addSettingCard(self.top_p_card)
        self.settings_group.addSettingCard(self.max_length_card)
        self.settings_group.addSettingCard(self.use_cache_card)
        self.settings_group.addSettingCard(self.iterative_prompt_card)
        self.settings_group.addSettingCard(self.temperature_card)

        self.setupLayout()
