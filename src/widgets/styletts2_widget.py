from PySide6.QtWidgets import QGroupBox, QHBoxLayout
from qfluentwidgets import FluentIcon as FIF

from src.widgets.generation_widget import GenerationWidget
from src.config.config import cfg
from ui.cards import RangeSettingCardScaled, RadioSettingCard


class StyleTTS2Widget(GenerationWidget):
    """
    Widget for StyleTTS2 text-to-speech generation.
    """
    def __init__(self, parent=None):
        super().__init__(text="StyleTTS2", parent=parent)
        
        self.text_input.setPlaceholderText("Please Enter Text")
        
        # Add StyleTTS2 specific settings
        self.style_weight_card = RangeSettingCardScaled(
            cfg.style_weight,
            FIF.PALETTE,
            self.tr('Style Weight'),
            self.tr('Controls how much of the reference style to apply'),
            parent=self
        )
        
        self.noise_scale_card = RangeSettingCardScaled(
            cfg.noise_scale,
            FIF.BRUSH,
            self.tr('Noise Scale'),
            self.tr('Controls the variation in the output'),
            parent=self
        )
        
        self.noise_scale_w_card = RangeSettingCardScaled(
            cfg.noise_scale_w,
            FIF.PALETTE,
            self.tr('Noise Scale W'),
            self.tr('Controls the variation in the output'),
            parent=self
        )
        
        self.length_scale_card = RangeSettingCardScaled(
            cfg.length_scale,
            FIF.SPEED_OFF,
            self.tr('Length Scale'),
            self.tr('Controls the speed of the output'),
            parent=self
        )
        
        self.noise_and_length = QGroupBox()
        self.noise_and_length.setStyleSheet("border: none")
        self.noise_and_length_layout = QHBoxLayout()
        self.noise_and_length_layout.setContentsMargins(0, 0, 0, 0)
        self.noise_and_length_layout.addWidget(self.noise_scale_card, 3)
        self.noise_and_length_layout.addWidget(self.length_scale_card, 3)
        self.noise_and_length.setLayout(self.noise_and_length_layout)
        
        self.style_and_noise_w = QGroupBox()
        self.style_and_noise_w.setStyleSheet("border: none")
        self.style_and_noise_w_layout = QHBoxLayout()
        self.style_and_noise_w_layout.setContentsMargins(0, 0, 0, 0)
        self.style_and_noise_w_layout.addWidget(self.style_weight_card, 3)
        self.style_and_noise_w_layout.addWidget(self.noise_scale_w_card, 3)
        self.style_and_noise_w.setLayout(self.style_and_noise_w_layout)
        
        self.slice_mode_card = RadioSettingCard(
            cfg.slice_mode,
            FIF.CUT,
            self.tr('Slice Mode'),
            self.tr('How to slice the sentence for longer TTS generation'),
            texts=["No Slice", "Basic punctuation . ! ? ...", "Every punctuation", "Every 4 sentences", "Every 2 sentences"],
            parent=self
        )
        
        # Insert StyleTTS2 specific settings before the RVC settings
        self.main_layout.insertWidget(self.main_layout.indexOf(self.rvc_enabled), self.style_and_noise_w)
        self.main_layout.insertWidget(self.main_layout.indexOf(self.rvc_enabled), self.noise_and_length)
        self.main_layout.insertWidget(self.main_layout.indexOf(self.rvc_enabled), self.slice_mode_card)