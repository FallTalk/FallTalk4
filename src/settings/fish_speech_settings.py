from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SettingCardGroup, RangeSettingCard, isDarkTheme, SwitchSettingCard
)
from qfluentwidgets import ScrollArea, ExpandLayout

from src.config.config import cfg
from src.ui.cards import RangeSettingCardScaled


class FishSpeechSettings(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr(''), self.scroll_widget)

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
        self.settings_group.addSettingCard(self.torch_compile_card)
        self.settings_group.addSettingCard(self.repetition_card)
        self.settings_group.addSettingCard(self.top_p_card)
        self.settings_group.addSettingCard(self.max_length_card)
        self.settings_group.addSettingCard(self.use_cache_card)
        self.settings_group.addSettingCard(self.iterative_prompt_card)
        self.settings_group.addSettingCard(self.seed_card)
        self.settings_group.addSettingCard(self.temperature_card)

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