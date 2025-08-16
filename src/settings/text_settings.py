from qfluentwidgets import (
    FluentIcon as FIF
)

from src.config.config import cfg
from src.settings.generic_settings import GenericSettings
from src.ui.cards import RangeSettingCardScaled


class TextSettings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.__initWidget()

    def __initWidget(self):
        self.setupLayout()

