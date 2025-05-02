from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QVBoxLayout, QSizePolicy, QSpacerItem
from qfluentwidgets import TextEdit, PrimaryPushButton, FluentIcon as FIF

from src.widgets.falltalk_widget import FallTalkWidget
from src.audio.audio_player import StandardAudioPlayerBar
from src.config.config import cfg
from src.ui.cards import TextSettingCard, SpinSettingCard, ComboBoxSettingsCard, ComboBoxWordsCard, RadioSettingCard, RangeSettingCardScaled

class GenerationWidget(FallTalkWidget):
    """
    Base widget class for text-to-speech generation interfaces.
    """
    def __init__(self, text=str, parent=None):
        super().__init__(text=text, parent=parent, vertical=True)
        
        self.text_input = TextEdit(self)
        self.text_input.setPlaceholderText("Please Enter Text")
        self.text_input.setFixedHeight(200)
        
        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVisible(False)
        
        self.main_layout.addWidget(self.text_input)
        self.addGenerationButton()
        self.addGenSettings()
        self.main_layout.addWidget(self.media_player)

    def addGenerationButton(self):
        """Add the generation button to the widget"""
        self.generate_button = PrimaryPushButton('Generate', self, FIF.PLAY)
        self.generate_button.clicked.connect(lambda: self.parent().generate_audio())
        
        self.button_layout = QHBoxLayout()
        self.button_layout.addStretch(1)
        self.button_layout.addWidget(self.generate_button)
        self.button_layout.addStretch(1)
        
        self.main_layout.addLayout(self.button_layout)

    def addGenSettings(self):
        """Add generation settings to the widget"""
        self.output_name = TextSettingCard(
            cfg.output_name,
            FIF.SAVE,
            self.tr('Output Name'),
            self.tr('Name of the output file'),
            parent=self
        )
        
        self.rvc_enabled = QGroupBox("RVC Post Processing")
        self.rvc_enabled.setCheckable(True)
        self.rvc_enabled.setChecked(cfg.get(cfg.rvc_enabled))
        self.rvc_enabled.toggled.connect(lambda x: cfg.set(cfg.rvc_enabled, x))
        self.rvc_enabled.setVisible(False)
        
        self.rvc_layout = QVBoxLayout()
        self.rvc_layout.setContentsMargins(0, 0, 0, 0)
        
        self.rvc_transpose = RangeSettingCardScaled(
            cfg.rvc_transpose,
            FIF.MUSIC,
            self.tr('Transpose'),
            self.tr('Pitch shift the output audio'),
            parent=self
        )
        
        self.rvc_layout.addWidget(self.rvc_transpose)
        self.rvc_enabled.setLayout(self.rvc_layout)
        
        self.addTempAndRep()
        
        self.main_layout.addWidget(self.output_name)
        self.main_layout.addWidget(self.rvc_enabled)
        
        # Add spacer at the bottom
        self.main_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def addTempAndRep(self):
        """Add temperature and repetition penalty settings"""
        self.temp_and_rep = QGroupBox()
        self.temp_and_rep.setStyleSheet("border: none")
        self.temp_and_rep_layout = QHBoxLayout()
        self.temp_and_rep_layout.setContentsMargins(0, 0, 0, 0)
        
        self.temperature_card = RangeSettingCardScaled(
            cfg.temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Controls the randomness of the generation'),
            parent=self
        )
        
        self.repetition_penalty_card = RangeSettingCardScaled(
            cfg.repetition_penalty,
            FIF.SPEED_OFF,
            self.tr('Repetition Penalty'),
            self.tr('Penalizes repetition in the generated text'),
            parent=self
        )
        
        self.temp_and_rep_layout.addWidget(self.temperature_card, 3)
        self.temp_and_rep_layout.addWidget(self.repetition_penalty_card, 3)
        self.temp_and_rep.setLayout(self.temp_and_rep_layout)