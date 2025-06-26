from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp


import os

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QStackedWidget, QSpacerItem, QFileDialog, QLineEdit
from qfluentwidgets import FluentIcon as FIF, TextEdit, PrimaryPushButton, SegmentedWidget, RangeSettingCard, SwitchSettingCard, ConfigValidator, ConfigItem, PushSettingCard, \
    Flyout, FlyoutView, FlyoutAnimationType, ToolButton


from src.config.config import cfg, FileValidator
from src.audio.audio_player import StandardAudioPlayerBar
from src.audio.audio_recorder import StandardAudioRecorderBar
from src.ui.cards import TextSettingCard, RvcComboBoxSettingsCard
from src.utils.inference_utils import get_edge_tts_voices, get_eleven_labs_voices
from src.utils.icons import FallTalkIcons
from src.widgets.falltalk_widget import FallTalkWidget
from src.enums.engine_type import EngineType
from src.settings.rvc_settings import RVCSettings
from src.widgets.drawer import RightDrawer
from src.help.rvc_help import RVCHelp


class BaseRVCWidget(QWidget):
    def __init__(self, parent: FallTalkApp):
        super().__init__(parent)
        self.parent = parent
        self.view = QVBoxLayout(self)
        self.view.setContentsMargins(0, 0, 0, 0)

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

        self.help_drawer = RightDrawer(self, title="About", icon=FIF.QUESTION)
        self.settings_drawer = RightDrawer(self, title="Advanced Settings", icon=FIF.SETTING)

        self.settings_button = ToolButton()
        self.settings_button.setIcon(FIF.SETTING)
        self.settings_button.setEnabled(True)
        self.settings_button.clicked.connect(lambda: self.toggle_settings_drawer())
        self.settings_button.setFixedWidth(50)

        self.help_button = ToolButton()
        self.help_button.setIcon(FIF.QUESTION)
        self.help_button.setEnabled(True)
        self.help_button.clicked.connect(lambda: self.toggle_help_drawer())
        self.help_button.setFixedWidth(50)

        self.settings_drawer.addWidget(RVCSettings(self))
        self.help_drawer.addWidget(RVCHelp(self))

    def show_settings(self, settings):
        view = FlyoutView(
            title='Advanced Settings',
            content="",
            icon=FIF.SETTING,
            parent=self,
            isClosable=True
        )

        # Add settings widget
        view.vBoxLayout.addWidget(settings)
        window_rect = self.window().geometry()

        # Adjust flyout size
        view.setMinimumWidth(max(1000, int(window_rect.width() * 0.75)))
        view.setMinimumHeight(max(600, int(window_rect.height() * 0.75)))

        # Calculate center point of the window
        view_size = view.sizeHint()

        # Calculate dynamic offsets based on window and view sizes
        x_offset = -(window_rect.width() * 0.3)  # Move left by 20% of window width
        y_offset = -(window_rect.height() * 0.2)  # Move up by 10% of window height

        center_point = QPoint(
            int(window_rect.x() + (window_rect.width() - view_size.width()) // 2 + x_offset),
            int(window_rect.y() + (window_rect.height() - view_size.height()) // 2 + y_offset)
        )

        # Show the flyout at the center point
        w = Flyout.make(view, center_point, self, aniType=FlyoutAnimationType.NONE)
        view.closed.connect(w.close)

    def toggle_settings_drawer(self):
        self.settings_drawer.open_drawer()

    def toggle_help_drawer(self):
        self.help_drawer.open_drawer()

class RVCMicrophoneWidget(BaseRVCWidget):

    def __init__(self, parent: FallTalkApp):
        super().__init__(parent)

        self.media_recorder = StandardAudioRecorderBar(self)
        self.view.addWidget(self.media_recorder)
        self.spacer = QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.view.addItem(self.spacer)
        self.addGenSettings()
        self.gen_settings_layout.addWidget(self.settings_button, stretch=1)
        self.gen_settings_layout.addWidget(self.help_button, stretch=1)

class RVCFileWidget(BaseRVCWidget):

    def __init__(self, parent: FallTalkApp):
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

        self.addGenSettings()

        # Add to layout
        self.buttons_layout = QHBoxLayout()
        self.generate_button = PrimaryPushButton(text="Generate Audio")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.generate_audio)
        self.buttons_layout.addWidget(self.generate_button, stretch=5)
        self.buttons_layout.addWidget(self.settings_button, stretch=1)
        self.buttons_layout.addWidget(self.help_button, stretch=1)
        self.view.addLayout(self.buttons_layout)

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

    def __init__(self, parent: FallTalkApp):
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

        self.addGenSettings()
        self.gen_settings_1_layout.addWidget(self.voice_combo, 3)

        # Add to layout
        self.buttons_layout = QHBoxLayout()
        self.generate_button = PrimaryPushButton(text="Generate Audio")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.generate_audio)
        self.buttons_layout.addWidget(self.generate_button, stretch=5)
        self.buttons_layout.addWidget(self.settings_button, stretch=1)
        self.buttons_layout.addWidget(self.help_button, stretch=1)
        self.view.addLayout(self.buttons_layout)

    def populate_voice_combo(self):
        if self.voice_combo.configItem.count() == 0:
            voices = get_edge_tts_voices()

            for voice in voices:
                self.voice_combo.configItem.addItem(voice)

            self.voice_combo.configItem.setCurrentIndex(self.voice_combo.configItem.count() - 1)


class RVCElevenLabsWidget(BaseRVCWidget):

    def __init__(self, parent: FallTalkApp):
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
        self.addGenSettings()
        self.gen_settings_1_layout.addWidget(self.voice_combo, 3)

        # Add to layout
        self.buttons_layout = QHBoxLayout()
        self.generate_button = PrimaryPushButton(text="Generate Audio")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.generate_audio)
        self.buttons_layout.addWidget(self.generate_button, stretch=5)
        self.buttons_layout.addWidget(self.settings_button, stretch=1)
        self.buttons_layout.addWidget(self.help_button, stretch=1)
        self.view.addLayout(self.buttons_layout)

    def populate_voice_combo(self):
        self.voice_combo.configItem.clear()
        voices = get_eleven_labs_voices()

        for voice in voices:
            self.voice_combo.configItem.addItem(voice)

        self.voice_combo.configItem.setCurrentIndex(self.voice_combo.configItem.count() - 1)


class RVCWidget(FallTalkWidget):

    def __init__(self, parent: FallTalkApp):
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
