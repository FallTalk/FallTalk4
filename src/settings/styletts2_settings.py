from qfluentwidgets import (
    RangeSettingCard
)

from src.config.config import cfg
from src.settings.generic_settings import GenericSettings
from src.ui.cards import RangeSettingCardScaled
from src.utils.icons import FallTalkIcons


class StyleTTS2Settings(GenericSettings):


    def __init__(self, parent=None):
        super().__init__(parent)

        self.alpha_card = RangeSettingCardScaled(
            cfg.style_alpha,
            FallTalkIcons.ALPHA.icon(stroke=True),
            self.tr('Alpha α'),
            self.tr("Timbre of speech. Lower values lean to the reference audio."),
            parent=self.settings_group
        )

        self.beta_card = RangeSettingCardScaled(
            cfg.style_beta,
            FallTalkIcons.BETA.icon(stroke=True),
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
        # add cards to group
        self.settings_group.addSettingCard(self.alpha_card)
        self.settings_group.addSettingCard(self.beta_card)
        self.settings_group.addSettingCard(self.diffusion_steps)
        self.settings_group.addSettingCard(self.embedding_scale)

        self.setupLayout()
