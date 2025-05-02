from PySide6.QtWidgets import QGroupBox, QHBoxLayout
from qfluentwidgets import FluentIcon as FIF, PrimaryPushButton

from src.config.config import cfg
from src.ui.cards import RadioSettingCard, RangeSettingCardScaled
from src.widgets.generation_widget import GenerationWidget


class GPT_SoVITSWidget(GenerationWidget):
    """
    Widget for GPT-SoVITS text-to-speech generation.
    """
    def __init__(self, parent=None):
        super().__init__(text="GPT SoVITS", parent=parent)
        
        self.text_input.setPlaceholderText("Please Select the 'Transcribe Reference Audio' button below")
        self.transcribe_state = None
        self.words_data = None
        
        # Add GPT-SoVITS specific settings
        self.mode_card = RadioSettingCard(
            cfg.slice_mode,
            FIF.CUT,
            self.tr('Slice Mode'),
            self.tr('How to slice the sentence for longer TTS generation'),
            texts=["No Slice", "Basic punctuation . ! ? ...", "Every punctuation", "Every 4 sentences", "Every 2 sentences"],
            parent=self
        )
        
        self.temperature_card = RangeSettingCardScaled(
            cfg.temperature_gpt_sovits,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Controls the randomness of the generation'),
            parent=self
        )
        
        self.speed_card = RangeSettingCardScaled(
            cfg.speed_gpt_sovits,
            FIF.SPEED_OFF,
            self.tr('Speed'),
            self.tr('Increase or decrease the generated audio speed'),
            parent=self
        )
        
        self.temp_and_speed = QGroupBox()
        self.temp_and_speed.setStyleSheet("border: none")
        self.temp_and_speed_layout = QHBoxLayout()
        self.temp_and_speed_layout.setContentsMargins(0, 0, 0, 0)
        self.temp_and_speed_layout.addWidget(self.temperature_card, 3)
        self.temp_and_speed_layout.addWidget(self.speed_card, 3)
        self.temp_and_speed.setLayout(self.temp_and_speed_layout)
        
        self.top_p_card = RangeSettingCardScaled(
            cfg.top_p_gpt_sovits,
            FIF.ALIGNMENT,
            self.tr('Top P'),
            self.tr('Controls the diversity of the generation'),
            parent=self
        )
        
        self.top_k_card = RangeSettingCardScaled(
            cfg.top_k_gpt_sovits,
            FIF.ALIGNMENT,
            self.tr('Top K'),
            self.tr('Controls the diversity of the generation'),
            parent=self
        )
        
        self.top_p_and_k = QGroupBox()
        self.top_p_and_k.setStyleSheet("border: none")
        self.top_p_and_k_layout = QHBoxLayout()
        self.top_p_and_k_layout.setContentsMargins(0, 0, 0, 0)
        self.top_p_and_k_layout.addWidget(self.top_p_card, 3)
        self.top_p_and_k_layout.addWidget(self.top_k_card, 3)
        self.top_p_and_k.setLayout(self.top_p_and_k_layout)
        
        # Add transcribe button
        self.transcribe_button = PrimaryPushButton('Transcribe Reference Audio', self, FIF.MICROPHONE)
        self.transcribe_button.clicked.connect(self.transcribe)
        self.transcribe_button.setEnabled(False)
        
        self.button_layout.insertWidget(1, self.transcribe_button)
        
        # Insert GPT-SoVITS specific settings before the RVC settings
        self.main_layout.insertWidget(self.main_layout.indexOf(self.rvc_enabled), self.mode_card)
        self.main_layout.insertWidget(self.main_layout.indexOf(self.rvc_enabled), self.temp_and_speed)
        self.main_layout.insertWidget(self.main_layout.indexOf(self.rvc_enabled), self.top_p_and_k)
        
        self.setVisible(cfg.engine.value == "GPT_SoVITS")
        self.media_player.setVisible(cfg.engine.value == "GPT_SoVITS")
        self.setEnabled(False)

    def onReferenceSelect(self):
        if self.parent().tts_engine and self.parent().tts_engine.is_base:
            self.generate_button.setEnabled(False)
            self.transcribe_button.setEnabled(True)
        else:
            self.generate_button.setEnabled(True)
            self.transcribe_button.setVisible(False)

    def transcribe(self):
        self.parent().transcribe(self)

    def clear(self):
        self.generate_button.setEnabled(False)
        self.transcribe_button.setEnabled(False)

    def load_data(self):
        self.transcribe_state = self.transcribe_state['transcript']
        self.text_input.setPlaceholderText(f'Transcript: {self.transcribe_state} \n\nPlease Enter Your Text Now')
        self.generate_button.setEnabled(True)
        self.transcribe_button.setEnabled(False)