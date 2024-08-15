from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SettingCardGroup, RangeSettingCard, isDarkTheme, SwitchSettingCard
)
from qfluentwidgets import ScrollArea, ExpandLayout

from falltalk.config import cfg, RangeSettingCardScaled, RadioSettingCard
from icons import FallTalkIcons


class GPTSoVITSSettings(ScrollArea):


    def __init__(self, parent=None):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr(''), self.scroll_widget)

        self.mode_card = RadioSettingCard(
            cfg.slice_mode,
            FIF.CUT,
            self.tr('Slice Mode'),
            self.tr('How to slice the sentence for longer TTS generation'),
            texts=["No Slice", "Slice by basic punct: . ! ? ...", "Slice by every punct", "Slice every 4 sentences", "Slice every 2 sentences"],
            parent=self.settings_group
        )

        self.low_vram_card = SwitchSettingCard(
            FallTalkIcons.RAM.icon(),
            self.tr('Low VRAM'),
            self.tr('Use FP16 precision, which uses less VRAM but is also less accurate'),
            configItem=cfg.low_vram_gpt_sovits,
            parent=self.settings_group
        )

        self.top_p_card = RangeSettingCardScaled(
            cfg.top_p_gpt_sovits,
            FIF.UP,
            self.tr('Top P'),
            self.tr('Higher values give more creativity in generation'),
            parent=self.settings_group
        )

        self.top_k_card = RangeSettingCard(
            cfg.top_k_gpt_sovits,
            FIF.UP,
            self.tr('Top K'),
            self.tr('Lower values make it more predictable and coherent'),
            parent=self.settings_group
        )

        self.temperature_card = RangeSettingCardScaled(
            cfg.temperature_gpt_sovits,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Recommend Keeping at 1'),
            parent=self.settings_group
        )

        self.speed_card = RangeSettingCardScaled(
            cfg.speed_gpt_sovits,
            FIF.SPEED_OFF,
            self.tr('Speed'),
            self.tr('Increase or decrease the generated audio speed'),
            parent=self.settings_group
        )

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
        self.settings_group.addSettingCard(self.mode_card)
        self.settings_group.addSettingCard(self.top_k_card)
        self.settings_group.addSettingCard(self.top_p_card)
        self.settings_group.addSettingCard(self.temperature_card)
        self.settings_group.addSettingCard(self.speed_card)

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
