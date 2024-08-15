from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SettingCardGroup, SwitchSettingCard, RangeSettingCard,
    isDarkTheme
)
from qfluentwidgets import ScrollArea, ExpandLayout

from falltalk.config import cfg, RangeSettingCardScaled
from icons import FallTalkIcons


class XTTSSettings(ScrollArea):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr(''), self.scroll_widget)

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

        # self.speed_card = RangeSettingCardScaled(
        #     cfg.speed,
        #     FIF.SPEED_OFF,
        #     title="Speed",
        #     parent=self.settings_group
        # )

        self.__initWidget()

    def __initWidget(self):
        self.resize(1000, 800)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 0, 0, 20)
        self.setWidget(self.scroll_widget)
        self.setWidgetResizable(True)

        # initialize style sheet
        self.__setQss()

        # initialize layout
        self.__initLayout()
        self.__connectSignalToSlot()

    def __initLayout(self):
        # add cards to group
        self.settings_group.addSettingCard(self.low_vram_card)
        self.settings_group.addSettingCard(self.deepspeed_card)
        self.settings_group.addSettingCard(self.temperature_card)
        self.settings_group.addSettingCard(self.repetition_penalty_card)
        # self.settings_group.addSettingCard(self.speed_card)

        # add setting card group to layout
        self.expand_layout.setSpacing(28)
        self.expand_layout.setContentsMargins(15, 0, 15, 0)
        self.expand_layout.addWidget(self.settings_group)

    def __setQss(self):
        """ set style sheet """
        self.scroll_widget.setObjectName('scrollWidget')

        theme = 'dark' if isDarkTheme() else 'light'
        with open(f'resource/qss/{theme}/setting_interface.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def __connectSignalToSlot(self):
        pass
