from PySide6.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QStackedWidget
from qfluentwidgets import FluentIcon as FIF, TextEdit, PrimaryPushButton, SegmentedWidget

from config.config import cfg
from src.audio.audio_player import StandardAudioPlayerBar
from src.audio.audio_recorder import StandardAudioRecorderBar
from src.ui.cards import TextSettingCard, ComboBoxSettingsCard, RangeSettingCardScaled
from src.widgets.falltalk_widget import FallTalkWidget


class BaseRVCWidget(QWidget):
    """
    Base widget for RVC (Real-time Voice Conversion) functionality.
    """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

    def addButtons(self):
        """Add generation buttons to the widget"""
        self.generate_button = PrimaryPushButton('Generate', self, FIF.PLAY)
        self.generate_button.clicked.connect(lambda: self.parent().parent().generate_audio())
        
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
        
        self.transpose_card = RangeSettingCardScaled(
            cfg.rvc_transpose,
            FIF.MUSIC,
            self.tr('Transpose'),
            self.tr('Pitch shift the output audio'),
            parent=self
        )
        
        self.index_rate_card = RangeSettingCardScaled(
            cfg.index_rate,
            FIF.ALIGNMENT,
            self.tr('Index Rate'),
            self.tr('Controls the quality of the output'),
            parent=self
        )
        
        self.filter_radius_card = RangeSettingCardScaled(
            cfg.filter_radius,
            FIF.BRUSH,
            self.tr('Filter Radius'),
            self.tr('Controls the quality of the output'),
            parent=self
        )
        
        self.rms_mix_rate_card = RangeSettingCardScaled(
            cfg.rms_mix_rate,
            FIF.ALIGNMENT,
            self.tr('RMS Mix Rate'),
            self.tr('Controls the volume of the output'),
            parent=self
        )
        
        self.protect_card = RangeSettingCardScaled(
            cfg.protect,
            FIF.SHIELD,
            self.tr('Protect'),
            self.tr('Protect the unvoiced consonants'),
            parent=self
        )
        
        self.index_and_filter = QGroupBox()
        self.index_and_filter.setStyleSheet("border: none")
        self.index_and_filter_layout = QHBoxLayout()
        self.index_and_filter_layout.setContentsMargins(0, 0, 0, 0)
        self.index_and_filter_layout.addWidget(self.index_rate_card, 3)
        self.index_and_filter_layout.addWidget(self.filter_radius_card, 3)
        self.index_and_filter.setLayout(self.index_and_filter_layout)
        
        self.rms_and_protect = QGroupBox()
        self.rms_and_protect.setStyleSheet("border: none")
        self.rms_and_protect_layout = QHBoxLayout()
        self.rms_and_protect_layout.setContentsMargins(0, 0, 0, 0)
        self.rms_and_protect_layout.addWidget(self.rms_mix_rate_card, 3)
        self.rms_and_protect_layout.addWidget(self.protect_card, 3)
        self.rms_and_protect.setLayout(self.rms_and_protect_layout)
        
        self.main_layout.addWidget(self.output_name)
        self.main_layout.addWidget(self.transpose_card)
        self.main_layout.addWidget(self.index_and_filter)
        self.main_layout.addWidget(self.rms_and_protect)

class RVCMicrophoneWidget(BaseRVCWidget):
    """
    Widget for RVC using microphone input.
    """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        
        self.media_recorder = StandardAudioRecorderBar(self)
        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVisible(False)
        
        self.main_layout.addWidget(self.media_recorder)
        self.addButtons()
        self.addGenSettings()
        self.main_layout.addWidget(self.media_player)

class RVCFileWidget(BaseRVCWidget):
    """
    Widget for RVC using file input.
    """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        
        self.text_input = TextEdit(self)
        self.text_input.setPlaceholderText("Please Select a File")
        self.text_input.setFixedHeight(200)
        self.text_input.setReadOnly(True)
        
        self.rvc_file = TextSettingCard(
            cfg.rvc_file,
            FIF.FOLDER,
            self.tr('RVC File'),
            self.tr('File to convert'),
            parent=self
        )
        self.rvc_file.clicked.connect(self.__onFileCardClicked)
        
        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVisible(False)
        
        self.main_layout.addWidget(self.text_input)
        self.main_layout.addWidget(self.rvc_file)
        self.addButtons()
        self.addGenSettings()
        self.main_layout.addWidget(self.media_player)

    def __onFileCardClicked(self):
        self.text_input.setPlainText(f"Selected File: {self.rvc_file.value}")

class RVCEdgeTTSWidget(BaseRVCWidget):
    """
    Widget for RVC using Edge TTS input.
    """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        
        self.text_input = TextEdit(self)
        self.text_input.setPlaceholderText("Please Enter Text")
        self.text_input.setFixedHeight(200)
        
        self.voice_combo = ComboBoxSettingsCard(
            cfg.edge_tts_voice,
            FIF.MICROPHONE,
            self.tr('Voice'),
            self.tr('Voice to use for Edge TTS'),
            parent=self
        )
        
        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVisible(False)
        
        self.main_layout.addWidget(self.text_input)
        self.main_layout.addWidget(self.voice_combo)
        self.addButtons()
        self.addGenSettings()
        self.main_layout.addWidget(self.media_player)
        
        self.populate_voice_combo()

    def populate_voice_combo(self):
        """Populate the voice combo box with available voices"""
        self.voice_combo.comboBox.clear()
        self.voice_combo.comboBox.addItems([
            "en-US-AnaNeural", "en-US-AriaNeural", "en-US-ChristopherNeural", 
            "en-US-EricNeural", "en-US-GuyNeural", "en-US-JennyNeural", 
            "en-US-MichelleNeural", "en-US-RogerNeural", "en-US-SteffanNeural"
        ])

class RVCElevenLabsWidget(BaseRVCWidget):
    """
    Widget for RVC using ElevenLabs TTS input.
    """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        
        self.text_input = TextEdit(self)
        self.text_input.setPlaceholderText("Please Enter Text")
        self.text_input.setFixedHeight(200)
        
        self.voice_combo = ComboBoxSettingsCard(
            cfg.eleven_labs_voice,
            FIF.MICROPHONE,
            self.tr('Voice'),
            self.tr('Voice to use for ElevenLabs'),
            parent=self
        )
        
        self.stability_card = RangeSettingCardScaled(
            cfg.eleven_labs_stability,
            FIF.ALIGNMENT,
            self.tr('Stability'),
            self.tr('Controls the stability of the generation'),
            parent=self
        )
        
        self.similarity_boost_card = RangeSettingCardScaled(
            cfg.eleven_labs_similarity_boost,
            FIF.ALIGNMENT,
            self.tr('Similarity Boost'),
            self.tr('Controls the similarity to the reference voice'),
            parent=self
        )
        
        self.stability_and_similarity = QGroupBox()
        self.stability_and_similarity.setStyleSheet("border: none")
        self.stability_and_similarity_layout = QHBoxLayout()
        self.stability_and_similarity_layout.setContentsMargins(0, 0, 0, 0)
        self.stability_and_similarity_layout.addWidget(self.stability_card, 3)
        self.stability_and_similarity_layout.addWidget(self.similarity_boost_card, 3)
        self.stability_and_similarity.setLayout(self.stability_and_similarity_layout)
        
        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVisible(False)
        
        self.main_layout.addWidget(self.text_input)
        self.main_layout.addWidget(self.voice_combo)
        self.main_layout.addWidget(self.stability_and_similarity)
        self.addButtons()
        self.addGenSettings()
        self.main_layout.addWidget(self.media_player)
        
        self.populate_voice_combo()

    def populate_voice_combo(self):
        """Populate the voice combo box with available voices"""
        self.voice_combo.comboBox.clear()
        self.voice_combo.comboBox.addItems([
            "Adam", "Antoni", "Arnold", "Bella", "Callum", "Charlie", "Clyde", 
            "Daniel", "Dorothy", "Ethan", "Fin", "Freya", "Gigi", "Giovanni", 
            "Grace", "Harry", "James", "Jeremy", "Jessie", "Joseph", "Josh", 
            "Liam", "Matthew", "Matilda", "Michael", "Mimi", "Nicole", "Patrick", 
            "Rachel", "Ryan", "Sam", "Sarah", "Scott", "Thomas", "Victoria"
        ])

class RVCWidget(FallTalkWidget):
    """
    Main widget for RVC functionality, containing all RVC sub-widgets.
    """
    def __init__(self, parent=None):
        super().__init__(text="RVC", parent=parent, vertical=True)
        
        self.segmented_widget = SegmentedWidget(self)
        self.segmented_widget.setObjectName("rvcSegmentedWidget")
        
        self.stackedWidget = QStackedWidget(self)
        self.stackedWidget.setObjectName("rvcStackedWidget")
        
        self.rvc_mic_widget = RVCMicrophoneWidget(self)
        self.rvc_file_widget = RVCFileWidget(self)
        self.edge_tts_widget = RVCEdgeTTSWidget(self)
        self.eleven_labs_widget = RVCElevenLabsWidget(self)
        
        self.addSubInterface(self.rvc_mic_widget, "rvcMicWidget", "Microphone")
        self.addSubInterface(self.rvc_file_widget, "rvcFileWidget", "File")
        self.addSubInterface(self.edge_tts_widget, "edgeTTSWidget", "Edge TTS")
        self.addSubInterface(self.eleven_labs_widget, "elevenLabsWidget", "ElevenLabs")
        
        self.segmented_widget.setCurrentItem("rvcMicWidget")
        self.segmented_widget.currentItemChanged.connect(self.onCurrentIndexChanged)
        
        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVisible(False)
        
        self.main_layout.addWidget(self.segmented_widget)
        self.main_layout.addWidget(self.stackedWidget)

    def onCurrentIndexChanged(self, index):
        """Handle change of current RVC tab"""
        if index == "rvcMicWidget":
            self.stackedWidget.setCurrentWidget(self.rvc_mic_widget)
        elif index == "rvcFileWidget":
            self.stackedWidget.setCurrentWidget(self.rvc_file_widget)
        elif index == "edgeTTSWidget":
            self.stackedWidget.setCurrentWidget(self.edge_tts_widget)
        elif index == "elevenLabsWidget":
            self.stackedWidget.setCurrentWidget(self.eleven_labs_widget)

    def addSubInterface(self, widget: QWidget, objectName, text):
        """Add sub interface to RVC widget"""
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.segmented_widget.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget)
        )