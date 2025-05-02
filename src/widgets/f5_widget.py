from PySide6.QtWidgets import QGroupBox, QHBoxLayout
from qfluentwidgets import FluentIcon as FIF, PrimaryPushButton

from src.config.config import cfg
from src.ui.cards import ComboBoxWordsCard, RadioSettingCard, RangeSettingCardScaled
from src.widgets.generation_widget import GenerationWidget


class F5Widget(GenerationWidget):
    """
    Widget for F5 text-to-speech generation.
    """
    def __init__(self, parent=None):
        super().__init__(text="F5", parent=parent)
        
        self.text_input.setPlaceholderText("Please Select the 'Transcribe Reference Audio' button below")
        self.transcribe_state = None
        self.words_data = None
        
        # Add F5 specific settings
        self.mode_card = RadioSettingCard(
            cfg.f5_mode,
            FIF.CUT,
            self.tr('Mode'),
            self.tr('Mode of operation'),
            texts=["tts", "edit"],
            parent=self
        )
        self.mode_card.radioButtonGroup.buttonClicked.connect(self.mode_changed)
        
        self.start_dropdown_card = ComboBoxWordsCard(
            cfg.f5_start_word,
            FIF.ALIGNMENT,
            self.tr('Start Word'),
            self.tr('Word to start editing from'),
            parent=self
        )
        
        self.end_dropdown_card = ComboBoxWordsCard(
            cfg.f5_end_word,
            FIF.ALIGNMENT,
            self.tr('End Word'),
            self.tr('Word to end editing at'),
            parent=self
        )
        
        self.start_end_group = QGroupBox()
        self.start_end_group.setStyleSheet("border: none")
        self.start_end_layout = QHBoxLayout()
        self.start_end_layout.setContentsMargins(0, 0, 0, 0)
        self.start_end_layout.addWidget(self.start_dropdown_card, 3)
        self.start_end_layout.addWidget(self.end_dropdown_card, 3)
        self.start_end_group.setLayout(self.start_end_layout)
        
        self.speed_card = RangeSettingCardScaled(
            cfg.speed_f5,
            FIF.SPEED_OFF,
            self.tr('Speed'),
            self.tr('Increase or decrease the generated audio speed'),
            parent=self
        )
        
        self.temperature_f5_card = RangeSettingCardScaled(
            cfg.temperature_f5,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Controls the randomness of the generation'),
            parent=self
        )
        
        self.temp_and_speed = QGroupBox()
        self.temp_and_speed.setStyleSheet("border: none")
        self.temp_and_speed_layout = QHBoxLayout()
        self.temp_and_speed_layout.setContentsMargins(0, 0, 0, 0)
        self.temp_and_speed_layout.addWidget(self.temperature_f5_card, 3)
        self.temp_and_speed_layout.addWidget(self.speed_card, 3)
        self.temp_and_speed.setLayout(self.temp_and_speed_layout)
        
        # Add transcribe button
        self.transcribe_button = PrimaryPushButton('Transcribe Reference Audio', self, FIF.MICROPHONE)
        self.transcribe_button.clicked.connect(self.transcribe)
        self.transcribe_button.setEnabled(False)
        
        self.button_layout.insertWidget(1, self.transcribe_button)
        
        # Insert F5 specific settings before the RVC settings
        self.main_layout.insertWidget(self.main_layout.indexOf(self.rvc_enabled), self.mode_card)
        self.main_layout.insertWidget(self.main_layout.indexOf(self.rvc_enabled), self.start_end_group)
        self.main_layout.insertWidget(self.main_layout.indexOf(self.rvc_enabled), self.temp_and_speed)
        
        # Hide edit-specific controls initially if mode is tts
        if cfg.get(cfg.f5_mode) == "tts":
            self.start_end_group.setVisible(False)
        
        self.setVisible(cfg.engine.value == "F5")
        self.media_player.setVisible(cfg.engine.value == "F5")
        self.setEnabled(False)

    def transcribe(self):
        self.parent().transcribe(self)

    def onReferenceSelect(self):
        if self.parent().tts_engine and self.parent().tts_engine.is_base:
            self.generate_button.setEnabled(False)
            self.transcribe_button.setEnabled(True)
        else:
            self.generate_button.setEnabled(True)
            self.transcribe_button.setVisible(False)

    def mode_changed(self, change):
        self.start_end_group.setVisible(cfg.get(cfg.f5_mode) == "edit")

    def clear(self):
        self.generate_button.setEnabled(False)
        self.transcribe_button.setEnabled(False)
        self.start_dropdown_card.comboBox.clear()
        self.end_dropdown_card.comboBox.clear()
        self.words_data = None

    def load_data(self):
        if self.transcribe_state and 'words' in self.transcribe_state:
            self.words_data = self.transcribe_state['words']
            self.start_dropdown_card.setWords(self.words_data)
            self.end_dropdown_card.setWords(self.words_data)
        
        self.transcribe_state = self.transcribe_state['transcript']
        self.text_input.setPlaceholderText(f'Transcript: {self.transcribe_state} \n\nPlease Enter Your Text Now')
        self.generate_button.setEnabled(True)
        self.transcribe_button.setEnabled(False)