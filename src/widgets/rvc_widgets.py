import os

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QStackedWidget, QSpacerItem, QFileDialog, QLineEdit
from qfluentwidgets import FluentIcon as FIF, TextEdit, PrimaryPushButton, SegmentedWidget, RangeSettingCard, SwitchSettingCard, ConfigValidator, ConfigItem, PushSettingCard, PushButton, Flyout, FlyoutView, FlyoutAnimationType

from src.config.config import cfg, FileValidator
from src.audio.audio_player import StandardAudioPlayerBar
from src.audio.audio_recorder import StandardAudioRecorderBar
from src.ui.cards import TextSettingCard, RangeSettingCardScaled, RvcComboBoxSettingsCard
from src.utils.inference_utils import get_edge_tts_voices, get_eleven_labs_voices
from src.utils.icons import FallTalkIcons
from src.widgets.falltalk_widget import FallTalkWidget
from src.enums.engine_type import EngineType
from src.settings.rvc_settings import RVCSettings


class BaseRVCWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.view = QVBoxLayout(self)
        self.view.setContentsMargins(0, 0, 0, 0)

    def addButtons(self):
        self.rvc_index_influence_card = RangeSettingCardScaled(
            cfg.rvc_index_influence,
            FIF.DICTIONARY,
            self.tr("Index Influence Ratio"),
            self.tr("Higher Values detail but risk artifacts. Increase until artifacts appear."),
        )

        self.rvc_filter_radius_card = RangeSettingCard(
            cfg.rvc_filter_radius,
            FIF.FILTER,
            self.tr("Filter Radius"),
            self.tr("Using median filtering on tones ≥ 3 can reduce respiration"),
        )

        self.train_infu = QGroupBox()
        self.train_infu.setStyleSheet("border: none")
        self.train_infu_layout = QHBoxLayout()
        self.train_infu_layout.setContentsMargins(0, 0, 0, 0)
        self.train_infu_layout.addWidget(self.rvc_filter_radius_card, 3)
        self.train_infu_layout.addWidget(self.rvc_index_influence_card, 3)
        self.train_infu.setLayout(self.train_infu_layout)
        self.view.addWidget(self.train_infu)

        self.rvc_autotune_card = SwitchSettingCard(
            FIF.MUSIC,
            self.tr("Autotune"),
            self.tr("Apply a soft autotune to your inferences, recommended signing."),
            configItem=cfg.rvc_autotune,
        )

        self.rvc_split_audio_card = SwitchSettingCard(
            FIF.CUT,
            self.tr("Split Audio"),
            self.tr("Split the audio into chunks for better results with large audio."),
            configItem=cfg.rvc_split_audio,
        )

        self.auto_and_split = QGroupBox()
        self.auto_and_split.setStyleSheet("border: none")
        self.auto_and_split_layout = QHBoxLayout()
        self.auto_and_split_layout.setContentsMargins(0, 0, 0, 0)
        self.auto_and_split_layout.addWidget(self.rvc_split_audio_card, 3)
        self.auto_and_split_layout.addWidget(self.rvc_autotune_card, 3)
        self.auto_and_split.setLayout(self.auto_and_split_layout)
        self.view.addWidget(self.auto_and_split)

    def addGenSettings(self):
        self.output_name = ConfigItem("TTS", "output_name", None, ConfigValidator())

        self.output_name_card = TextSettingCard(
            self.output_name,
            FIF.SAVE_AS,
            self.tr('Output Name'),
            self.tr('Name of Generated WAV file'),
            placeholder="Random"
        )

        self.gen_settings_1 = QGroupBox()
        self.gen_settings_1.setStyleSheet("border: none")
        self.gen_settings_1_layout = QHBoxLayout()
        self.gen_settings_1_layout.setContentsMargins(0, 0, 0, 0)
        self.gen_settings_1_layout.addWidget(self.output_name_card, 3)
        self.gen_settings_1.setLayout(self.gen_settings_1_layout)
        self.view.addWidget(self.gen_settings_1)

        self.autoplay = SwitchSettingCard(
            FIF.PLAY,
            self.tr('Autoplay'),
            self.tr('Automatically Play Generated Audio'),
            cfg.auto_play,
        )
        self.delete_leftovers = SwitchSettingCard(
            FIF.DELETE,
            self.tr('Keep Only FUZ'),
            self.tr('Delete XMW, LIP, and WAV'),
            cfg.keep_only_fuz
        )
        self.xwm_card = SwitchSettingCard(
            FIF.COMMAND_PROMPT,
            self.tr('Create FUZ'),
            self.tr('Create XWM, LIP, and FUZ'),
            cfg.xwm_enabled,
        )
        self.gen_settings = QGroupBox()
        self.gen_settings.setStyleSheet("border: none")
        self.gen_settings_layout = QHBoxLayout()
        self.gen_settings_layout.setContentsMargins(0, 0, 0, 0)
        self.gen_settings_layout.addWidget(self.autoplay, 2)
        self.gen_settings_layout.addWidget(self.xwm_card, 2)
        self.gen_settings_layout.addWidget(self.delete_leftovers, 2)

        self.gen_settings.setLayout(self.gen_settings_layout)
        self.view.addWidget(self.gen_settings)

        self.generate_button = PrimaryPushButton("Generate Audio")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.generate_audio)

        # Add settings button
        self.settings_button = PushButton("Settings")
        self.settings_button.setIcon(FIF.SETTING)
        self.settings_button.setEnabled(True)
        self.settings_button.setFixedWidth(100)
        self.settings_button.clicked.connect(lambda: self.show_settings(RVCSettings(self)))
        
        # Add to layout
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(self.settings_button, stretch=1)
        self.buttons_layout.addWidget(self.generate_button, stretch=5)
        self.view.addLayout(self.buttons_layout)

    def show_settings(self, settings):
        view = FlyoutView(
            title='RVC Settings',
            content="",
            icon=FIF.SETTING,
            parent=self,
            isClosable=True
        )

        # Add settings widget
        view.vBoxLayout.addWidget(settings)

        # Adjust flyout size
        screen_rect = self.window().screen().availableGeometry()
        width = min(1000, screen_rect.width() - 100)  # Leave some margin
        view.setMinimumWidth(width)

        # Calculate center point of the window
        window_rect = self.window().geometry()
        view_size = view.sizeHint()
        center_point = QPoint(
            window_rect.x() + window_rect.width() // 2 - view_size.width() // 2,
            window_rect.y() + window_rect.height() // 2 - view_size.height() // 2
        )

        # Show the flyout at the center point
        w = Flyout.make(view, center_point, self, aniType=FlyoutAnimationType.NONE)
        view.closed.connect(w.close)


class RVCMicrophoneWidget(BaseRVCWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.media_recorder = StandardAudioRecorderBar(self)
        self.view.addWidget(self.media_recorder)
        self.spacer = QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.view.addItem(self.spacer)
        self.rvc_pitch_card = RangeSettingCard(
            cfg.rvc_pitch,
            FIF.MARKET,
            self.tr("Pitch Adjustment"),
            self.tr("Set the pitch of the audio, useful for opposite gender."),
        )
        self.view.addWidget(self.rvc_pitch_card)
        self.addButtons()
        self.addGenSettings()


class RVCFileWidget(BaseRVCWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.rvc_file_start = "./"
        self.rvc_file = ConfigItem("bulk", "upload_file", "Please Select an Audio File", FileValidator())
        self.rvc_file_card = PushSettingCard(
            self.tr('Select File'),
            FIF.DOCUMENT,
            self.tr("Audio File"),
            self.rvc_file.value,
        )
        self.rvc_file_card.clicked.connect(self.__onFileCardClicked)
        self.view.addWidget(self.rvc_file_card)
        self.spacer = QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.view.addItem(self.spacer)

        self.rvc_pitch_card = RangeSettingCard(
            cfg.rvc_pitch,
            FIF.MARKET,
            self.tr("Pitch Adjustment"),
            self.tr("Set the pitch of the audio, useful for opposite gender."),
        )
        self.view.addWidget(self.rvc_pitch_card)

        self.addButtons()
        self.addGenSettings()

    def __onFileCardClicked(self):
        allowed_file_types = "WAV files (*.wav);;MP3 files (*.mp3)"
        folder = QFileDialog.getOpenFileName(
            self, self.tr("Choose CSV or Text File"), self.rvc_file_start, allowed_file_types)
        if not folder or folder[0] == "":
            return

        self.rvc_file_start = os.path.dirname(folder[0])
        self.rvc_file.value = folder[0]
        self.rvc_file_card.setContent(folder[0])


class RVCEdgeTTSWidget(BaseRVCWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.text_input = TextEdit()
        font = QFont()
        font.setPointSize(12)
        self.text_input.setFont(font)
        self.text_input.setPlaceholderText("Edge TTS, offered by Microsoft, is a free service that boasts a diverse array of voices and supports numerous languages. However, it currently lacks the capability to infuse emotional nuances into the synthesized speech.")
        self.view.addWidget(self.text_input)

        self.voice_combo = RvcComboBoxSettingsCard(
            FallTalkIcons.VOICE_OVER.icon(),
            self.tr('Voice'),
            self.tr('Which base voice should we use?'))

        self.addButtons()
        self.addGenSettings()
        self.gen_settings_1_layout.addWidget(self.voice_combo, 3)

    def populate_voice_combo(self):
        if self.voice_combo.configItem.count() == 0:
            voices = get_edge_tts_voices()

            for voice in voices:
                self.voice_combo.configItem.addItem(voice)

            self.voice_combo.configItem.setCurrentIndex(self.voice_combo.configItem.count() - 1)


class RVCElevenLabsWidget(BaseRVCWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.text_input = TextEdit()
        font = QFont()
        font.setPointSize(12)
        self.text_input.setFont(font)
        self.text_input.setPlaceholderText("ElevenLabs requires that you set an API key below. Once you have done that, you gain the ability to utilize all the voices on the ElevenLabs platform including ones you have created.")
        self.view.addWidget(self.text_input)

        self.voice_combo = RvcComboBoxSettingsCard(
            FallTalkIcons.VOICE_OVER.icon(),
            self.tr('Voice'),
            self.tr('Which base voice should we use?'))

        self.eleven_labs_key = TextSettingCard(
            cfg.rvc_eleven_labs_key,
            FIF.SAVE_AS,
            self.tr('API Access Key'),
            self.tr('Optional, used to access your custom ElevenLabs voices'),
            placeholder="Required to use service"
        )
        self.eleven_labs_key.lineEdit.setEchoMode(QLineEdit.EchoMode.Password)

        self.view.addWidget(self.eleven_labs_key)
        self.addButtons()
        self.addGenSettings()
        self.gen_settings_1_layout.addWidget(self.voice_combo, 3)

    def populate_voice_combo(self):
        self.voice_combo.configItem.clear()
        voices = get_eleven_labs_voices()

        for voice in voices:
            self.voice_combo.configItem.addItem(voice)

        self.voice_combo.configItem.setCurrentIndex(self.voice_combo.configItem.count() - 1)


class RVCWidget(FallTalkWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent, text="RVC", vertical=True)
        self.parent = parent

        self.pivot = SegmentedWidget(self)
        self.stackedWidget = QStackedWidget(self)

        self.edge_tts_widget = RVCEdgeTTSWidget(self.parent)
        self.eleven_labs_widget = RVCElevenLabsWidget(self.parent)
        self.rvc_file_widget = RVCFileWidget(self.parent)
        self.rvc_mic_widget = RVCMicrophoneWidget(self.parent)

        self.addSubInterface(self.rvc_mic_widget, 'rvc_mic_widget', 'Microphone')
        self.addSubInterface(self.rvc_file_widget, 'rvc_file_widget', 'File')
        self.addSubInterface(self.edge_tts_widget, 'edge_tts_widget', 'Edge TTS')
        self.addSubInterface(self.eleven_labs_widget, 'eleven_labs_widget', 'ElevenLabs')

        self.boxLayout.addWidget(self.pivot, 0, Qt.AlignmentFlag.AlignLeft)
        self.boxLayout.addWidget(self.stackedWidget)
        self.stackedWidget.currentChanged.connect(self.onCurrentIndexChanged)
        self.stackedWidget.setCurrentWidget(self.rvc_mic_widget)
        self.pivot.setCurrentItem(self.rvc_mic_widget.objectName())

        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVolume(100)
        self.addToFrame(self.media_player)
        self.setEnabled(False)
        self.setVisible(cfg.engine.value == EngineType.RVC.value)
        self.media_player.setVisible(cfg.engine.value == EngineType.RVC.value)

    def onCurrentIndexChanged(self, index):
        widget = self.stackedWidget.widget(index)
        self.pivot.setCurrentItem(widget.objectName())

        if index == 3:
            self.eleven_labs_widget.populate_voice_combo()
        elif index == 2:
            self.edge_tts_widget.populate_voice_combo()

    def addSubInterface(self, widget: QWidget, objectName, text):
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget)
        )
