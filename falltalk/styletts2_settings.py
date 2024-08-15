from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SettingCardGroup, RangeSettingCard, isDarkTheme, SwitchSettingCard, FluentIcon
)
from qfluentwidgets import ScrollArea, ExpandLayout

from falltalk.config import cfg, RangeSettingCardScaled, RadioSettingCard
from icons import FallTalkIcons, FallTalkStrokeIcons


class StyleTTS2Settings(ScrollArea):


    def __init__(self, parent=None):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr(''), self.scroll_widget)

        self.alpha_card = RangeSettingCardScaled(
            cfg.style_alpha,
            FallTalkStrokeIcons.ALPHA.icon(),
            self.tr('Alpha α'),
            self.tr("Timbre of speech. Lower values lean to the reference audio."),
            parent=self.settings_group
        )

        self.beta_card = RangeSettingCardScaled(
            cfg.style_beta,
            FallTalkStrokeIcons.BETA.icon(),
            self.tr('Beta β'),
            self.tr('Rhythm, stress, and intonation of speech. Lower values give more influence by the reference audio.'),
            parent=self.settings_group
        )

        self.diffusion_steps = RangeSettingCard(
            cfg.style_diffusion_steps,
            FallTalkIcons.STEPS.icon(),
            self.tr('Diffusion steps'),
            self.tr('A higher number of steps can lead to more refined results but increased processing time.'),
            parent=self.settings_group
        )

        self.embedding_scale = RangeSettingCard(
            cfg.style_embedding_scale,
            FallTalkIcons.SCALE.icon(),
            self.tr('Embedding Scale'),
            self.tr('Degree of emotion in the speech. Higher values result in more pronounced emotional expression.'),
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
        self.settings_group.addSettingCard(self.alpha_card)
        self.settings_group.addSettingCard(self.beta_card)
        self.settings_group.addSettingCard(self.diffusion_steps)
        self.settings_group.addSettingCard(self.embedding_scale)

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
