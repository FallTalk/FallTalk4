from PySide6.QtWidgets import QGroupBox, QHBoxLayout
from qfluentwidgets import FluentIcon as FIF, RangeSettingCard

from src.widgets.generation_widget import GenerationWidget
from src.config.config import cfg
from src.ui.cards import RangeSettingCardScaled, RadioSettingCard
from src.utils.icons import FallTalkStrokeIcons
from src.enums.engine_type import EngineType


class StyleTTS2Widget(GenerationWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent, text="StyleTTS2")

        self.alpha_card = RangeSettingCardScaled(
            cfg.style_alpha,
            FallTalkStrokeIcons.ALPHA.icon(),
            self.tr('Alpha α'),
            self.tr("Timbre of speech. Lower values lean to the reference audio."),
        )

        self.beta_card = RangeSettingCardScaled(
            cfg.style_beta,
            FallTalkStrokeIcons.BETA.icon(),
            self.tr('Beta β'),
            self.tr('Rhythm, stress, and intonation of speech. Lower values give more influence by the reference audio.'),
        )

        self.alpha_and_beta = QGroupBox()
        self.alpha_and_beta.setStyleSheet("border: none")
        self.alpha_and_beta_layout = QHBoxLayout()
        self.alpha_and_beta_layout.setContentsMargins(0, 0, 0, 0)
        self.alpha_and_beta_layout.addWidget(self.alpha_card, 3)
        self.alpha_and_beta_layout.addWidget(self.beta_card, 3)
        self.alpha_and_beta.setLayout(self.alpha_and_beta_layout)
        self.addToFrame(self.alpha_and_beta)

        self.diffusion_steps = RangeSettingCard(
            cfg.style_diffusion_steps,
            FIF.UP,
            self.tr('Diffusion steps'),
            self.tr('A higher number of steps can lead to more refined results but increased processing time.'),
        )

        self.embedding_scale = RangeSettingCard(
            cfg.style_embedding_scale,
            FIF.UP,
            self.tr('Embedding Scale'),
            self.tr('Degree of emotion in the speech. Higher values result in more pronounced emotional expression.'),
        )

        self.diffuse_and_embed = QGroupBox()
        self.diffuse_and_embed.setStyleSheet("border: none")
        self.diffuse_and_embed_layout = QHBoxLayout()
        self.diffuse_and_embed_layout.setContentsMargins(0, 0, 0, 0)
        self.diffuse_and_embed_layout.addWidget(self.embedding_scale, 3)
        self.diffuse_and_embed_layout.addWidget(self.diffusion_steps, 3)
        self.diffuse_and_embed.setLayout(self.diffuse_and_embed_layout)
        self.addToFrame(self.diffuse_and_embed)

        self.addGenSettings()
        self.text_input.setPlaceholderText("Please enter text")
        self.addGenerationButton()
        self.setVisible(cfg.engine.value == EngineType.STYLE_TTS2.value)
        self.media_player.setVisible(cfg.engine.value == EngineType.STYLE_TTS2.value)
        self.setEnabled(False)