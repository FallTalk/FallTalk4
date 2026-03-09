from qfluentwidgets import (
    FluentIcon as FIF, OptionsSettingCard
)

from src.config.config import cfg
from src.settings.generic_settings import GenericSettings
from src.ui.cards import TextSettingCard


class QwenSettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.model_version_card = OptionsSettingCard(
            cfg.qwen_model_version,
            FIF.TILES,
            self.tr('Model Version'),
            self.tr('Select the Qwen3 TTS model size/version'),
            texts=["1.7B-Base", "0.6B-Base"],
            parent=self.settings_group
        )

        self.language_card = OptionsSettingCard(
            cfg.qwen_language,
            FIF.LANGUAGE,
            self.tr('Language'),
            self.tr('Language for Qwen3 TTS'),
            texts=["Auto", "Chinese", "English", "Japanese", "Korean"],
            parent=self.settings_group
        )

        self.instruct_card = TextSettingCard(
            cfg.qwen_instruct,
            FIF.CHAT,
            self.tr('Instruct'),
            self.tr('Instruction for the speaker (e.g., "Very happy.")'),
            placeholder="Type Instruction Here",
            parent=self.settings_group
        )

        self.__initWidget()

    def __initWidget(self):
        self.settings_group.addSettingCard(self.model_version_card)
        self.settings_group.addSettingCard(self.language_card)
        self.settings_group.addSettingCard(self.instruct_card)

        self.setupLayout()
