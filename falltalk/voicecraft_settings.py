from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SettingCardGroup, RangeSettingCard, isDarkTheme
)
from qfluentwidgets import ScrollArea, ExpandLayout

from falltalk.config import cfg, RangeSettingCardScaled
from icons import FallTalkIcons


class VoiceCraftSettings(ScrollArea):


    def __init__(self, parent=None):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr(''), self.scroll_widget)

        self.stop_repetition_card = RangeSettingCard(
            cfg.stop_repetition,
            FallTalkIcons.LOOP.icon(),
            self.tr('Stop Repetition'),
            self.tr('If Long Pauses, change to 2 or 1. -1 = disabled'),
            parent=self.settings_group
        )

        self.sample_batch_size_card = RangeSettingCard(
            cfg.sample_batch_size,
            FIF.TILES,
            self.tr('Sample Batch Size'),
            self.tr('The higher the number, the faster the output will be. Under the hood, the model will generate this many samples and choose the shortest one.'),
            parent=self.settings_group
        )

        self.seed_card = RangeSettingCard(
            cfg.seed,
            FIF.LEAF,
            self.tr('Seed'),
            self.tr('-1 is always random'),
            parent=self.settings_group
        )

        self.kvcache_card = RangeSettingCard(
            cfg.kvcache,
            FallTalkIcons.RAM.icon(),
            self.tr('VRAM Cache'),
            self.tr('set to 0 to use less VRAM, but with slower inference'),
            parent=self.settings_group
        )

        self.left_margin_card = RangeSettingCardScaled(
            cfg.left_margin,
            FIF.LEFT_ARROW,
            self.tr('Left Margin'),
            self.tr('margin to the left of the editing segment'),
            parent=self.settings_group,
            scale=1000.0
        )

        self.right_margin_card = RangeSettingCardScaled(
            cfg.right_margin,
            FIF.RIGHT_ARROW,
            self.tr('Right Margin'),
            self.tr('margin to the right of the editing segment'),
            parent=self.settings_group,
            scale=1000.0
        )

        self.top_p_card = RangeSettingCardScaled(
            cfg.top_p,
            FIF.UP,
            self.tr('Top P'),
            self.tr('0.9 is a good value, 0.8 is also good'),
            parent=self.settings_group
        )

        # self.temperature_card = RangeSettingCardScaled(
        #     cfg.voicecraft_temperature,
        #     FIF.FRIGID,
        #     self.tr('Temperature'),
        #     self.tr('Recommend Keeping at 1'),
        #     parent=self.settings_group
        # )

        # self.top_k_card = RangeSettingCardScaled(
        #     cfg.top_k,
        #     FIF.SETTING,
        #     self.tr('Top K'),
        #     self.tr('0 means we don not use topk sampling, because we use topp sampling'),
        #     parent=self.settings_group
        # )
        #
        # self.codec_audio_sr_card = RangeSettingCard(
        #     cfg.codec_audio_sr,
        #     FIF.SETTING,
        #     self.tr('Codec Audio SR'),
        #     self.tr('encodec specific, Do not change'),
        #     parent=self.settings_group
        # )
        #
        # self.codec_sr_card = RangeSettingCardScaled(
        #     cfg.codec_sr,
        #     FIF.SETTING,
        #     self.tr('Codec SR'),
        #     self.tr('encodec specific, do not change'),
        #     parent=self.settings_group
        # )
        #
        # self.silence_tokens_card = OptionsSettingCard(
        #     cfg.silence_tokens,
        #     FIF.SETTING,
        #     self.tr('Silence Tokens'),
        #     self.tr('encodec specific, do not change'),
        #     texts=["[1388, 1898, 131]"],
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
        #self.settings_group.addSettingCard(self.mode_card)
        self.settings_group.addSettingCard(self.stop_repetition_card)
        self.settings_group.addSettingCard(self.sample_batch_size_card)
        self.settings_group.addSettingCard(self.seed_card)
        self.settings_group.addSettingCard(self.kvcache_card)
        self.settings_group.addSettingCard(self.left_margin_card)
        self.settings_group.addSettingCard(self.right_margin_card)
        self.settings_group.addSettingCard(self.top_p_card)
        #self.settings_group.addSettingCard(self.temperature_card)
        #self.settings_group.addSettingCard(self.top_k_card)
        #self.settings_group.addSettingCard(self.codec_audio_sr_card)
        #self.settings_group.addSettingCard(self.codec_sr_card)
        #self.settings_group.addSettingCard(self.silence_tokens_card)

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
