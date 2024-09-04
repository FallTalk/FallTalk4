import csv
import json
import os
import re
import shutil
from typing import Union, List

import soundfile as sf
import torch
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QModelIndex, QAbstractTableModel, QTimer, QUrl
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QStackedWidget, QWidget, QHeaderView, QGroupBox, QAbstractItemView, QSpacerItem, QSizePolicy, QFileDialog, QDialog, QLabel, QLineEdit
from qfluentwidgets import FluentIcon as FIF, TextEdit, PushButton, SegmentedWidget, \
    SearchLineEdit, RangeSettingCard, PrimaryPushButton, SwitchSettingCard, \
    ConfigItem, ConfigValidator, TableView, CheckBox, FluentIconBase, CommandBar, Action, TransparentDropDownPushButton, setFont, CheckableMenu, MenuIndicatorType, qrouter, FluentTitleBar, NavigationInterface, NavigationItemPosition, NavigationTreeWidget, BodyLabel, IconWidget, Theme, isDarkTheme, \
    FolderValidator, PushSettingCard, Dialog, MessageBoxBase, BoolValidator, RangeConfigItem, RangeValidator, OptionsConfigItem, OptionsValidator, TextWrap, SystemTrayMenu
from qfluentwidgets.components.dialog_box.dialog import Ui_MessageBox, MessageBox
from qfluentwidgets.components.widgets.combo_box import ComboItem
from qfluentwidgets.window.fluent_window import FluentWindowBase

from audio_player import StandardAudioPlayerBar
from audio_recorder import StandardAudioRecorderBar
from falltalk import falltalkutils
from falltalk.config import RangeSettingCardScaled, cfg, TextSettingCard, RadioSettingCard, ComboBoxSettingsCard, RvcComboBoxSettingsCard, FileValidator, CUSTOM_DISCLAIMER, CustomFolderValidator, SpinSettingCard, ComboBoxWordsCard
from falltalk.main_settings import FallTalkSettings
from falltalk.rvc_settings import RVCSettings
from falltalk.voicecraft_settings import VoiceCraftSettings
from falltalk.xtts_settings import XTTSSettings
from faq import FAQPage
from gpt_sovits_settings import GPTSoVITSSettings
from icons import FallTalkIcons, FallTalkStrokeIcons
from styletts2_settings import StyleTTS2Settings
from qframelesswindow import FramelessDialog


class CustomCommandBar(CommandBar):

    def __init__(self, parent=None):
        super().__init__(parent)

    def _visibleWidgets(self) -> List[QWidget]:
        """ return the visible widgets in layout """
        # have enough spacing to show all widgets
        return self._widgets


class FallTalkFluentWindow(FluentWindowBase):
    """ Fluent window """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitleBar(FluentTitleBar(self))

        self.navigationInterface = NavigationInterface(self, showReturnButton=True)

        self.toolbar = QHBoxLayout()
        self.toolbar.stretch(1)
        self.widgetLayout = QVBoxLayout()
        self.label = BodyLabel(self.tr("Character Models"))
        self.label.setFixedWidth(125)

        self.character_label = BodyLabel(self.tr("Please Load Model"))
        self.character_label.setFixedWidth(150)

        self.reference_label = BodyLabel(self.tr("Reference:"))
        self.reference_time_label = BodyLabel(self.tr("00:00"))
        self.reference_time_label.setFixedWidth(45)

        self.rvc_action = Action(FallTalkStrokeIcons.VOICE_SQUARE.icon(), self.tr('RVC'), checkable=True, checked=cfg.get(cfg.engine) == 'RVC')
        self.gpt_sovits_action = Action(FallTalkIcons.G.icon(), self.tr('GPT_SoVITS'), checkable=True, checked=cfg.get(cfg.engine) == 'GPT_SoVITS')
        self.voicecraft_action = Action(FallTalkIcons.VOICE.icon(), self.tr('VoiceCraft'), checkable=True, checked=cfg.get(cfg.engine) == 'VoiceCraft')
        self.xtts_action = Action(FallTalkIcons.FROG.icon(), self.tr('XTTSv2'), checkable=True, checked=cfg.get(cfg.engine) == 'XTTSv2')
        self.styletts2_action = Action(FallTalkIcons.STYLE.icon(), self.tr('StyleTTS2'), checkable=True, checked=cfg.get(cfg.engine) == 'StyleTTS2')
        self.cpu_action = Action(FallTalkIcons.CPU.icon(), self.tr('CPU'), checkable=True, checked=cfg.get(cfg.device) == 'cpu')
        self.gpu_action = Action(FallTalkIcons.GPU.icon(), self.tr('GPU'), checkable=True, checked=cfg.get(cfg.device) == 'cuda')
        self.gpu2_action = Action(FallTalkIcons.GPU.icon(), self.tr('GPU 2'), checkable=True, checked=cfg.get(cfg.device) == 'cuda:1')
        self.clean_action = Action(FIF.BROOM, self.tr('Clean'))

        self.update_action = Action(FallTalkIcons.IMPORTANT.icon(color=cfg.get(cfg.themeColor)), self.tr('Update Available'))
        self.new_models_action = Action(FallTalkIcons.NEW.icon(color=cfg.get(cfg.themeColor)), self.tr('New Models Added'))

        # initialize layout
        self.toolbar_1 = self.createCommandBar()
        self.toolbar.addWidget(self.toolbar_1, stretch=1)

        self.widgetLayout.addLayout(self.toolbar)

        self.hBoxLayout.addWidget(self.navigationInterface)
        self.hBoxLayout.addLayout(self.widgetLayout)
        self.hBoxLayout.setStretchFactor(self.widgetLayout, 1)

        self.widgetLayout.addWidget(self.stackedWidget)
        self.widgetLayout.setContentsMargins(0, 48, 0, 0)

        self.navigationInterface.displayModeChanged.connect(self.titleBar.raise_)
        self.titleBar.raise_()

    def _onCurrentInterfaceChanged(self, index: int):
        super()._onCurrentInterfaceChanged(index)
        self.label.setText(self.stackedWidget.currentWidget().title)

    def addSubInterface(self, interface: QWidget, icon: Union[FluentIconBase, QIcon, str], text: str,
                        position=NavigationItemPosition.TOP, parent=None, isTransparent=False) -> NavigationTreeWidget:

        if not interface.objectName():
            raise ValueError("The object name of `interface` can't be empty string.")
        if parent and not parent.objectName():
            raise ValueError("The object name of `parent` can't be empty string.")

        interface.setProperty("isStackedTransparent", isTransparent)
        self.stackedWidget.addWidget(interface)

        # add navigation item
        routeKey = interface.objectName()
        item = self.navigationInterface.addItem(
            routeKey=routeKey,
            icon=icon,
            text=text,
            onClick=lambda: self.switchTo(interface),
            position=position,
            tooltip=text,
            parentRouteKey=parent.objectName() if parent else None
        )

        # initialize selected item
        if self.stackedWidget.count() == 1:
            self.stackedWidget.currentChanged.connect(self._onCurrentInterfaceChanged)
            self.navigationInterface.setCurrentItem(routeKey)
            qrouter.setDefaultRouteKey(self.stackedWidget, routeKey)

        self._updateStackedBackground()

        return item

    def resizeEvent(self, e):
        self.titleBar.move(46, 0)
        self.titleBar.resize(self.width() - 46, self.titleBar.height())

    def createEngineMenu(self, pos=None):
        menu = CheckableMenu(parent=self, indicatorType=MenuIndicatorType.RADIO)
        menu.addActions([
            self.rvc_action,
            self.gpt_sovits_action,
            self.voicecraft_action,
            self.xtts_action,
            self.styletts2_action,
        ])
        if pos is not None:
            menu.exec(pos, ani=True)
        return menu

    def createDeviceMenu(self, pos=None):
        menu = CheckableMenu(parent=self, indicatorType=MenuIndicatorType.RADIO)
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                menu.addActions([
                    self.cpu_action,
                    self.gpu_action,
                    self.gpu2_action,
                    self.clean_action
                ])
            else:
                menu.addActions([
                    self.cpu_action,
                    self.gpu_action,
                    self.clean_action
                ])
        else:
            menu.addActions([
                self.cpu_action
            ])
        if pos is not None:
            menu.exec(pos, ani=True)
        return menu

    def createCommandBar_2(self):
        bar = CustomCommandBar(self)
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        # add custom widget

        return bar

    def __reset(self):
        if cfg.get(cfg.engine) == "XTTSv2":
            cfg.resetXtts()
        elif cfg.get(cfg.engine) == "VoiceCraft":
            cfg.resetVoiceCraft()
        elif cfg.get(cfg.engine) == "GPT_SoVITS":
            cfg.resetGPT()
        elif cfg.get(cfg.engine) == "StyleTTS2":
            cfg.resetStyleTTS()

        cfg.resetRvc()
        cfg.resetMusicAndFX()

    def createCommandBar(self):
        bar = CommandBar(self)

        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        bar.addWidget(self.label)
        bar.addSeparator()
        bar.addWidget(self.character_label)
        bar.addSeparator()
        bar.addWidget(self.reference_label)
        bar.addWidget(self.reference_time_label)
        bar.addSeparator()
        reset = Action(FIF.ROTATE, self.tr('Reset Generation Settings'))
        bar.addActions([reset])
        bar.addSeparator()
        reset.triggered.connect(self.__reset)
        button = TransparentDropDownPushButton(self.tr('Engine'), self, FIF.DEVELOPER_TOOLS)
        button.setMenu(self.createEngineMenu())
        button.setFixedHeight(34)
        setFont(button, 12)
        bar.addWidget(button)
        bar.addSeparator()

        cpu_button = TransparentDropDownPushButton(self.tr('Device'), self, FallTalkIcons.GPU.icon())
        cpu_button.setMenu(self.createDeviceMenu())
        cpu_button.setFixedHeight(34)
        setFont(cpu_button, 12)
        bar.addWidget(cpu_button)

        # spacer = QWidget()
        # spaceLayout = QVBoxLayout()
        # spaceLayout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        # spacer.setLayout(spaceLayout)
        # bar.addWidget(StretchingWidget())

        # bar.addActions([
        #     Action(FIF.ADD, self.tr('Add')),
        #     Action(FIF.ROTATE, self.tr('Rotate')),
        #     Action(FIF.ZOOM_IN, self.tr('Zoom in')),
        #     Action(FIF.ZOOM_OUT, self.tr('Zoom out')),
        # ])
        # bar.addSeparator()
        # bar.addActions([
        #     Action(FIF.EDIT, self.tr('Edit'), checkable=True),
        #     Action(FIF.INFO, self.tr('Info')),
        #     Action(FIF.DELETE, self.tr('Delete')),
        #     Action(FIF.SHARE, self.tr('Share'))
        # ])

        # add custom widget
        # button = TransparentDropDownPushButton(self.tr('Sort'), self, FIF.SCROLL)
        # button.setMenu(self.createCheckableMenu())
        # button.setFixedHeight(34)
        # setFont(button, 12)
        # bar.addWidget(button)

        # bar.addHiddenActions([
        #     Action(FIF.SETTING, self.tr('Settings'), shortcut='Ctrl+I'),
        # ])
        return bar


class FallTalkWidget(QFrame):

    def __init__(self, text: str, parent=None, vertical=False):
        super().__init__(parent=parent)
        if vertical:
            self.boxLayout = QVBoxLayout(self)
        else:
            self.boxLayout = QHBoxLayout(self)
        self.title = text
        # self.settingLabel = SubtitleLabel(self.tr(text), self)
        # self.settingLabel.move(6, 5)
        # self.settingLabel.setFixedWidth(400)

        self.setObjectName(text.replace(' ', '-'))
        # !IMPORTANT: leave some space for title bar
        self.boxLayout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(self.boxLayout)

    def addToFrame(self, widget):
        self.boxLayout.addWidget(widget)


class FaqWidget(FallTalkWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent, text="FAQ", vertical=True)
        self.faq = FAQPage(parent)
        self.addToFrame(self.faq)


class SettingsWidget(FallTalkWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent, text="Settings", vertical=True)

        # Create a TabView instance
        self.pivot = SegmentedWidget(self)
        self.stackedWidget = QStackedWidget(self)

        # Add TabItems to the TabView
        self.rvc_settings = RVCSettings(parent)
        self.voicecraft_settings = VoiceCraftSettings(parent)
        self.xtts_settings = XTTSSettings(parent)
        self.gpt_sovits_settings = GPTSoVITSSettings(parent)
        self.styletts2_settings = StyleTTS2Settings(parent)

        self.engine_settings = FallTalkSettings(parent)

        # add items to pivot
        self.addSubInterface(self.engine_settings, 'main_settings', 'Main Settings')
        self.addSubInterface(self.rvc_settings, 'rvc_settings', 'RVC')
        self.addSubInterface(self.voicecraft_settings, 'voicecraft_settings', 'VoiceCraft')
        self.addSubInterface(self.xtts_settings, 'xtts_settings', 'XTTS')
        self.addSubInterface(self.gpt_sovits_settings, 'gpt_sovits_settings', 'GPT SoVITS')
        self.addSubInterface(self.styletts2_settings, 'styletts2_settings', 'StyleTTS2')

        self.boxLayout.addWidget(self.pivot, 0, Qt.AlignmentFlag.AlignLeft)
        self.boxLayout.addWidget(self.stackedWidget)

        self.stackedWidget.currentChanged.connect(self.onCurrentIndexChanged)
        self.stackedWidget.setCurrentWidget(self.engine_settings)
        self.pivot.setCurrentItem(self.engine_settings.objectName())

    def addSubInterface(self, widget: QWidget, objectName, text):
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget)
        )

    def onCurrentIndexChanged(self, index):
        widget = self.stackedWidget.widget(index)
        self.pivot.setCurrentItem(widget.objectName())


class GenerationWidget(FallTalkWidget):

    def __init__(self, text=str, parent=None):
        super().__init__(parent=parent, text=text, vertical=True)
        self.text_input = TextEdit()
        font = QFont()
        font.setPointSize(12)
        self.text_input.setFont(font)
        self.addToFrame(self.text_input)
        self.parent = parent
        self.gen_settings = None
        self.gen_settings2 = None
        self.media_player = None
        self.generate_button = None

    def addGenerationButton(self):
        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVolume(100)
        self.generate_button = PrimaryPushButton("Generate Audio")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.generate_audio)
        self.addToFrame(self.generate_button)
        self.addToFrame(self.media_player)

    def addGenSettings(self):
        self.output_name = ConfigItem("TTS", "output_name", None, ConfigValidator())

        self.autoplay = SwitchSettingCard(
            FIF.PLAY,
            self.tr('Autoplay'),
            self.tr('Automatically Play Generated Audio'),
            cfg.auto_play,
        )
        self.output_name_card = TextSettingCard(
            self.output_name,
            FIF.SAVE_AS,
            self.tr('Output Name'),
            self.tr('Name of Generated WAV file'),
            placeholder="Random"
        )
        self.xwm_card = SwitchSettingCard(
            FIF.COMMAND_PROMPT,
            self.tr('Create FUZ'),
            self.tr('Create XWM, LIP, and FUZ'),
            cfg.xwm_enabled,
        )
        self.rvc_enabled = SwitchSettingCard(
            FIF.MEGAPHONE,
            self.tr('RVC'),
            self.tr('Use RVC Upscaler (Recommended)'),
            cfg.rvc_enabled
        )

        self.delete_leftovers = SwitchSettingCard(
            FIF.DELETE,
            self.tr('Keep Only FUZ'),
            self.tr('Delete XMW, LIP, and WAV'),
            cfg.keep_only_fuz
        )

        self.gen_settings = QGroupBox()
        self.gen_settings.setStyleSheet("border: none")
        self.gen_settings_layout = QHBoxLayout()
        self.gen_settings_layout.setContentsMargins(0, 0, 0, 0)

        self.gen_settings_layout.addWidget(self.autoplay, 2)
        self.gen_settings_layout.addWidget(self.rvc_enabled, 2)
        self.gen_settings.setLayout(self.gen_settings_layout)

        self.gen_settings2 = QGroupBox()
        self.gen_settings2.setStyleSheet("border: none")
        self.gen_settings2_layout = QHBoxLayout()
        self.gen_settings2_layout.setContentsMargins(0, 0, 0, 0)

        self.gen_settings2_layout.addWidget(self.xwm_card, 2)
        self.gen_settings2_layout.addWidget(self.delete_leftovers, 2)
        self.gen_settings2.setLayout(self.gen_settings2_layout)

        self.addToFrame(self.output_name_card)
        self.addToFrame(self.gen_settings2)
        self.addToFrame(self.gen_settings)

    def addTempAndRep(self):
        self.temperature_card = RangeSettingCardScaled(
            cfg.model_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Randomness, 1 = balanced, 0 = disabled'),
        )
        self.repetition_penalty_card = RangeSettingCard(
            cfg.model_repetition,
            FallTalkIcons.LOOP.icon(),
            self.tr('Repetition Penalty'),
            self.tr('Discourage same sounds during generation'),
        )
        self.temp_and_rep = QGroupBox()
        self.temp_and_rep.setStyleSheet("border: none")
        self.temp_and_rep_layout = QHBoxLayout()
        self.temp_and_rep_layout.setContentsMargins(0, 0, 0, 0)
        self.temp_and_rep_layout.addWidget(self.repetition_penalty_card, 3)
        self.temp_and_rep_layout.addWidget(self.temperature_card, 3)
        self.temp_and_rep.setLayout(self.temp_and_rep_layout)
        self.addToFrame(self.temp_and_rep)

    def addTempAndStopRep(self):
        # self.temperature_card = RangeSettingCardScaled(
        #     cfg.voicecraft_temperature,
        #     FIF.FRIGID,
        #     self.tr('Temperature'),
        #     self.tr('Randomness, 1 = balanced, 0 = disabled'),
        # )
        self.repetition_penalty_card = RangeSettingCard(
            cfg.stop_repetition,
            FallTalkIcons.LOOP.icon(),
            self.tr('Stop Repetition'),
            self.tr('If Long Pauses, change to 2 or 1. -1 = disabled'),
        )
        self.temp_and_rep = QGroupBox()
        self.temp_and_rep.setStyleSheet("border: none")
        self.temp_and_rep_layout = QHBoxLayout()
        self.temp_and_rep_layout.setContentsMargins(0, 0, 0, 0)
        self.temp_and_rep_layout.addWidget(self.repetition_penalty_card, 3)
        # self.temp_and_rep_layout.addWidget(self.temperature_card, 3)
        self.temp_and_rep.setLayout(self.temp_and_rep_layout)
        self.addToFrame(self.temp_and_rep)


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
        self.view.addWidget(self.generate_button)


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
            voices = falltalkutils.get_edge_tts_voices()

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
        voices = falltalkutils.get_eleven_labs_voices()

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
        self.setVisible(cfg.engine.value == "RVC")
        self.media_player.setVisible(cfg.engine.value == "RVC")

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


class XttsWidget(GenerationWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent, text="XTTS")
        self.addTempAndRep()
        self.addGenSettings()
        self.text_input.setPlaceholderText("Please enter text")
        self.addGenerationButton()
        self.setVisible(cfg.engine.value == "XTTSv2")
        self.media_player.setVisible(cfg.engine.value == "XTTSv2")
        self.setEnabled(False)


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
        self.setVisible(cfg.engine.value == "StyleTTS2")
        self.media_player.setVisible(cfg.engine.value == "StyleTTS2")
        self.setEnabled(False)


class GPT_SoVITSWidget(GenerationWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent, text="GPT SoVITS")
        self.text_input.setPlaceholderText("Please Select the 'Transcribe Reference Audio' button below")
        self.transcribe_state = None
        self.words_data = None

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
            FIF.UP,
            self.tr('Top P'),
            self.tr('Higher values give more creativity in generation.'),
            parent=self
        )

        self.top_k_card = RangeSettingCard(
            cfg.top_k_gpt_sovits,
            FIF.UP,
            self.tr('Top K'),
            self.tr('Lower values make it more predictable and coherent'),
            parent=self
        )

        self.addToFrame(self.mode_card)
        self.addToFrame(self.temp_and_speed)

        self.p_and_k = QGroupBox()
        self.p_and_k.setStyleSheet("border: none")
        self.p_and_k_layout = QHBoxLayout()
        self.p_and_k_layout.setContentsMargins(0, 0, 0, 0)
        self.p_and_k_layout.addWidget(self.top_p_card, 3)
        self.p_and_k_layout.addWidget(self.top_k_card, 3)
        self.p_and_k.setLayout(self.p_and_k_layout)
        self.addToFrame(self.p_and_k)

        self.addGenSettings()

        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVolume(100)
        self.buttons_layout = QHBoxLayout()
        self.transcribe_button = PrimaryPushButton("Transcribe Reference Audio")
        self.transcribe_button.setIcon(FIF.PENCIL_INK)
        self.transcribe_button.clicked.connect(self.transcribe)
        self.transcribe_button.setVisible(False)
        self.buttons_layout.addWidget(self.transcribe_button, stretch=1)
        self.generate_button = PrimaryPushButton("Generate Audio")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.generate_audio)
        self.buttons_layout.addWidget(self.generate_button, stretch=1)
        self.boxLayout.addLayout(self.buttons_layout)
        self.addToFrame(self.media_player)

        self.setVisible(cfg.engine.value == "GPT_SoVITS")
        self.media_player.setVisible(cfg.engine.value == "GPT_SoVITS")
        self.setEnabled(False)

    def onReferenceSelect(self):
        if self.parent.tts_engine and self.parent.tts_engine.is_base:
            self.generate_button.setEnabled(False)
            self.transcribe_button.setEnabled(True)
        elif self.parent.tts_engine:
            self.generate_button.setEnabled(True)
            self.transcribe_button.setVisible(False)

    def transcribe(self):
        self.parent.transcribe(self)

    def clear(self):
        self.generate_button.setEnabled(False)
        self.transcribe_button.setEnabled(False)

    def load_data(self):
        self.transcribe_state = self.transcribe_state['transcript']
        self.text_input.setPlaceholderText(f'Transcript: {self.transcribe_state} \n\nPlease Enter Your Text Now')
        self.generate_button.setEnabled(True)
        self.transcribe_button.setEnabled(False)


class VoiceCraftWidget(GenerationWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent, text="VoiceCraft")
        self.text_input.setPlaceholderText("Please Select the 'Transcribe Reference Audio' button below")
        self.transcribe_state = None
        self.words_data = None

        self.mode_card = RadioSettingCard(
            cfg.mode,
            FIF.DEVELOPER_TOOLS,
            self.tr('Generation Mode'),
            self.tr('How should we generate text'),
            texts=["Edit", "TTS", "Long TTS"],
        )

        self.mode_card.optionChanged.connect(self.mode_changed)
        # self.edit_mode_card = RadioSettingCard(
        #     cfg.edit_mode,
        #     FIF.SETTING,
        #     self.tr('Editing Mode'),
        #     self.tr('What to do with the selected first and last word'),
        #     texts=["Replace Half", "Replace Completely"],
        # )

        self.addToFrame(self.mode_card)

        self.start_dropdown_card = ComboBoxWordsCard(
            FIF.RIGHT_ARROW,
            self.tr('Start'),
            self.tr('Where do we start generating the new text'))
        self.end_dropdown_card = ComboBoxWordsCard(
            FIF.LEFT_ARROW,
            self.tr('End'),
            self.tr('Where do we stop generating the new text'))

        self.start_and_end = QGroupBox()
        self.start_and_end.setStyleSheet("border: none")
        self.start_and_end_layout = QHBoxLayout()
        self.start_and_end_layout.setContentsMargins(0, 0, 0, 0)
        self.start_and_end_layout.addWidget(self.start_dropdown_card, 3)
        self.start_and_end_layout.addWidget(self.end_dropdown_card, 3)
        self.start_and_end.setLayout(self.start_and_end_layout)
        self.start_and_end.setVisible(cfg.get(cfg.mode) == "edit")
        # self.edit_mode_card.setVisible(cfg.get(cfg.mode) == "edit")
        self.addToFrame(self.start_and_end)

        self.start_tts_dropdown_card = ComboBoxWordsCard(
            FIF.RIGHT_ARROW,
            self.tr('Start'),
            self.tr('Where do we start generating the new text'))

        self.start_tts = QGroupBox()
        self.start_tts.setStyleSheet("border: none")
        self.start_tts_layout = QHBoxLayout()
        self.start_tts_layout.setContentsMargins(0, 0, 0, 0)
        self.start_tts_layout.addWidget(self.start_tts_dropdown_card)
        self.start_tts.setLayout(self.start_tts_layout)
        self.start_tts.setVisible(cfg.get(cfg.mode) == "tts" or cfg.get(cfg.mode) == "long_tts")
        self.addToFrame(self.start_tts)
        # self.addToFrame(self.edit_mode_card)
        self.addTempAndStopRep()
        self.addGenSettings()

        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVolume(100)
        self.buttons_layout = QHBoxLayout()
        self.transcribe_button = PrimaryPushButton("Transcribe Reference Audio")
        self.transcribe_button.setIcon(FIF.PENCIL_INK)
        self.transcribe_button.clicked.connect(self.transcribe)
        self.transcribe_button.setEnabled(False)
        self.buttons_layout.addWidget(self.transcribe_button, stretch=1)
        self.generate_button = PrimaryPushButton("Generate Audio")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.generate_audio)
        self.generate_button.setEnabled(False)
        self.buttons_layout.addWidget(self.generate_button, stretch=1)
        self.boxLayout.addLayout(self.buttons_layout)
        self.addToFrame(self.media_player)

        self.setVisible(cfg.engine.value == "VoiceCraft")
        self.media_player.setVisible(cfg.engine.value == "VoiceCraft")
        self.setEnabled(True)

    def transcribe(self):
        self.parent.transcribe(self)

    def onReferenceSelect(self):
        self.generate_button.setEnabled(False)
        self.transcribe_button.setEnabled(True)

    def mode_changed(self, change):
        self.start_tts.setVisible(change.value == "tts" or change.value == "long_tts")
        self.start_and_end.setVisible(change.value == "edit")
        # self.edit_mode_card.setVisible(change.value == "edit")

    def clear(self):
        self.generate_button.setEnabled(False)
        self.transcribe_button.setEnabled(False)
        self.start_dropdown_card.configItem.clear()
        self.end_dropdown_card.configItem.clear()
        self.start_tts_dropdown_card.configItem.clear()
        self.start_tts_dropdown_card.configItem.clear()
        self.start_dropdown_card.setTranscript(None)
        self.end_dropdown_card.setTranscript(None)
        self.start_tts_dropdown_card.setTranscript(None)

    def load_data(self):
        self.generate_button.setEnabled(True)
        self.transcribe_button.setEnabled(False)
        self.text_input.setPlaceholderText(f"Transcript: {self.transcribe_state['transcript']} \n\nPlease Enter Your Text Now")
        falltalkutils.logger.debug(f"{self.transcribe_state['words_info']}")
        self.start_dropdown_card.setTranscript(self.transcribe_state['words_info'])
        self.end_dropdown_card.setTranscript(self.transcribe_state['words_info'])
        self.start_tts_dropdown_card.setTranscript(self.transcribe_state['words_info'])

        for word_info in self.transcribe_state['words_info']:
            self.start_dropdown_card.configItem.addItem(f"{word_info['word']}\t{word_info['start']}", userData=word_info)
            self.end_dropdown_card.configItem.addItem(f"{word_info['word']}\t{word_info['end']}", userData=word_info)
            self.start_tts_dropdown_card.configItem.addItem(f"{word_info['word']}\t{word_info['end']}", userData=word_info)

        self.end_dropdown_card.configItem.setCurrentIndex(self.end_dropdown_card.configItem.count() - 1)
        self.start_tts_dropdown_card.configItem.setCurrentIndex(self.end_dropdown_card.configItem.count() - 1)


class TableModel(QAbstractTableModel):

    def __init__(self, data, headers, parent=None):
        super().__init__(parent)
        self._data = data
        self._headers = headers

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return self._data[index.row()][index.column()]

        return None

    def getData(self):
        return self._data

    def full_data(self, row, column):
        return self._data[row][column]

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]
        return super().headerData(section, orientation, role)

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role == Qt.ItemDataRole.EditRole:
            self._data[index.row()][index.column()] = value
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
            return True
        return False

    def flags(self, index):
        return Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable


class CharacterTableModel(QAbstractTableModel):

    def __init__(self, data, headers, parent=None):
        super().__init__(parent)
        self._data = sorted(data, key=lambda x: x[1])
        self._headers = headers

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole):
        if role == Qt.ItemDataRole.DisplayRole and (index.column() == 1 or index.column() == 2):
            return self._data[index.row()][index.column()]

        return None

    def full_data(self, row, column):
        return self._data[row][column]

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]
        return super().headerData(section, orientation, role)


class CustomMessageBox(MessageBoxBase):
    """ Custom message box """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = QLabel(f'Import {cfg.get(cfg.engine)} Model', self)
        self.viewLayout.insertWidget(0, self.titleLabel, 0, Qt.AlignmentFlag.AlignTop)

        self.ckpt_file = ConfigItem("custom", "ckpt_file", "Please Select a 'ckpt' File", FileValidator(allowed_file_types=['.pth']))

        self.ckpt_file_card = PushSettingCard(
            self.tr('Select .ckpt File'),
            FIF.DOCUMENT,
            self.tr("Checkpoint File"),
            self.ckpt_file.value,
        )

        self.pth_file = ConfigItem("custom", "pth_file", "Please Select a 'pth' File", FileValidator(allowed_file_types=['.pth']))

        self.pth_file_card = PushSettingCard(
            self.tr('Select .pth File'),
            FIF.DOCUMENT,
            self.tr("Path File"),
            self.pth_file.value,
        )

        self.index_file = ConfigItem("custom", "index_file", "Please Select an 'index' File", FileValidator(allowed_file_types=['.index']))

        self.index_file_card = PushSettingCard(
            self.tr('Select .index File'),
            FIF.DOCUMENT,
            self.tr("Index file"),
            self.index_file.value,
        )

        self.custom_name = ConfigItem("custom", "custom_name", None, ConfigValidator())

        self.custom_name_card = TextSettingCard(
            self.custom_name,
            FIF.SAVE_AS,
            self.tr('Name'),
            self.tr('Name of Custom Model'),
            placeholder="Required"
        )

        self.start_dir = "./"

        # add widget to view layout
        self.viewLayout.addWidget(self.custom_name_card)
        self.viewLayout.addWidget(self.ckpt_file_card)
        self.viewLayout.addWidget(self.index_file_card)
        self.viewLayout.addWidget(self.pth_file_card)

        self.index_file_card.setVisible(cfg.get(cfg.engine) == 'RVC')
        self.ckpt_file_card.setVisible(cfg.get(cfg.engine) == 'GPT_SoVITS')

        self.setMinimumWidth(600)

        self.yesButton.setDisabled(True)

        self.pth_file_card.clicked.connect(self.__onPathCardClicked)
        self.index_file_card.clicked.connect(self.__onIndexCardClicked)
        self.ckpt_file_card.clicked.connect(self.__onCkptCardClicked)
        self.custom_name_card.lineEdit.textChanged.connect(self.enableYesButton)
        self.custom_name_card.lineEdit.textEdited.connect(self.enableYesButton)

    def enableYesButton(self):
        if self.custom_name.value and self.custom_name.value != '':
            if cfg.get(cfg.engine) == 'RVC' and self.index_file.value != "Please Select an 'index' File" and self.pth_file.value != "Please Select a 'pth' File":
                self.yesButton.setEnabled(True)
            elif cfg.get(cfg.engine) == 'GPT_SoVITS' and self.ckpt_file.value != "Please Select a 'ckpt' File" and self.pth_file.value != "Please Select a 'pth' File":
                self.yesButton.setEnabled(True)
            elif not cfg.get(cfg.engine) == 'GPT_SoVITS' and not cfg.get(cfg.engine) == 'RVC' and self.pth_file.value != "Please Select a 'pth' File":
                self.yesButton.setEnabled(True)
            else:
                self.yesButton.setEnabled(False)

    def __onIndexCardClicked(self):
        allowed_file_types = "Index File (*.index)"
        folder = QFileDialog.getOpenFileName(
            self, self.tr("Select Index File"), self.start_dir, allowed_file_types)
        if not folder or folder[0] == "":
            return

        self.index_file.value = folder[0]
        self.start_dir = os.path.dirname(folder[0])
        self.index_file_card.setContent(folder[0])
        self.enableYesButton()

    def __onPathCardClicked(self):
        allowed_file_types = "Path File (*.pth)"
        folder = QFileDialog.getOpenFileName(
            self, self.tr("Select Path File"), self.start_dir, allowed_file_types)
        if not folder or folder[0] == "":
            return

        self.pth_file.value = folder[0]
        self.start_dir = os.path.dirname(folder[0])
        self.pth_file_card.setContent(folder[0])
        self.enableYesButton()

    def __onCkptCardClicked(self):
        allowed_file_types = "Checkpoint File (*.ckpt)"
        folder = QFileDialog.getOpenFileName(
            self, self.tr("Select Checkpoint File"), self.start_dir, allowed_file_types)
        if not folder or folder[0] == "":
            return

        self.ckpt_file.value = folder[0]
        self.start_dir = os.path.dirname(folder[0])
        self.ckpt_file_card.setContent(folder[0])
        self.enableYesButton()


class CharactersWidget(FallTalkWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent, text="Character Models", vertical=True)
        self.parent = parent
        # Create a TabView instance
        self.pivot = SegmentedWidget(self)
        self.stackedWidget = QStackedWidget(self)
        self.trained_table = TableView()
        self.trained_table.setBorderVisible(True)
        self.trained_table.setBorderRadius(8)
        self.trained_table.setAlternatingRowColors(True)
        self.trained_table.setWordWrap(False)
        self.trained_table.verticalHeader().setVisible(False)
        headers = ["Load", "Name", "Directory", "Update", "Delete", "RVC"]
        model = CharacterTableModel([], headers)
        self.trained_table.setModel(model)
        self.trained_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.trained_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.trained_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.trained_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.trained_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.trained_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.trained_table.setSortingEnabled(True)
        self.trained_table.setColumnWidth(5, 40)
        self.trained_table.setColumnWidth(4, 125)
        self.trained_table.setColumnWidth(3, 125)
        self.trained_table.setColumnWidth(0, 125)

        self.untrained_table = TableView()
        self.untrained_table.setBorderVisible(True)
        self.untrained_table.setBorderRadius(8)
        self.untrained_table.setAlternatingRowColors(True)
        self.untrained_table.verticalHeader().setVisible(False)
        self.untrained_table.setModel(CharacterTableModel([], ["Load", "Name", "Directory", "RVC"]))
        self.untrained_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.untrained_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.untrained_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.untrained_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.untrained_table.setSortingEnabled(True)
        self.untrained_table.setColumnWidth(3, 40)
        self.untrained_table.setColumnWidth(0, 125)

        self.custom_widget = QWidget()
        self.custom_widget.setContentsMargins(0, 0, 0, 0)

        self.custom_view = QVBoxLayout(self.custom_widget)
        self.custom_view.setContentsMargins(0, 0, 0, 0)
        self.custom_table = TableView()
        self.custom_table.setBorderVisible(True)
        self.custom_table.setBorderRadius(8)
        self.custom_table.setAlternatingRowColors(True)
        self.custom_table.verticalHeader().setVisible(False)
        self.custom_table.setModel(CharacterTableModel([], ["Load", "Name", "Directory", "Delete", "RVC"]))
        self.custom_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.custom_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.custom_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.custom_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.custom_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.custom_table.setSortingEnabled(True)
        self.custom_table.setColumnWidth(4, 40)
        self.custom_table.setColumnWidth(0, 125)
        self.custom_table.setColumnWidth(3, 125)

        # self.stackedWidget.currentChanged.connect(self.verify)

        self.add_button = PushButton("Add")
        self.add_button.setMaximumWidth(200)
        self.add_button.clicked.connect(self.add_custom)
        self.add_button.setIcon(FIF.ADD_TO)

        self.custom_view.addWidget(self.custom_table)
        self.custom_view.addWidget(self.add_button)

        self.filter_line_edit = SearchLineEdit()
        self.filter_line_edit.setPlaceholderText("Filter...")
        self.filter_line_edit.textChanged.connect(self.apply_filter)
        self.controlsBox = QHBoxLayout()
        self.rvc_checkbox = CheckBox("RVC Only")
        self.rvc_checkbox.setMinimumWidth(200)
        self.rvc_checkbox.stateChanged.connect(self.apply_filter)
        self.controlsBox.addWidget(self.filter_line_edit)
        self.controlsBox.addWidget(self.rvc_checkbox)

        # add items to pivot
        self.addSubInterface(self.trained_table, 'trained_table', 'Trained')
        self.addSubInterface(self.untrained_table, 'untrained_table', 'Untrained')
        self.addSubInterface(self.custom_widget, 'custom_table', 'Custom')

        self.boxLayout.addWidget(self.pivot, 0, Qt.AlignmentFlag.AlignLeft)
        self.boxLayout.addWidget(self.stackedWidget)
        self.boxLayout.addLayout(self.controlsBox)

        self.stackedWidget.currentChanged.connect(self.onCurrentIndexChanged)
        self.stackedWidget.setCurrentWidget(self.trained_table)
        self.pivot.setCurrentItem(self.trained_table.objectName())

    def onCurrentIndexChanged(self, index):
        widget = self.stackedWidget.widget(index)
        self.pivot.setCurrentItem(widget.objectName())

    def verify(self, obj):
        if obj == 2 and not cfg.get(cfg.accepts_custom_disclaimer):
            title = 'Disclaimer for Use for Custom Imports'
            content = CUSTOM_DISCLAIMER
            w = MessageBox(title, content, self.parent.window())
            w.yesButton.setText(self.tr('Agree'))
            if w.exec():
                cfg.set(cfg.accepts_custom_disclaimer, True)

    def add_custom(self):
        if cfg.get(cfg.accepts_custom_disclaimer):
            m = CustomMessageBox(self.parent.window())
            if m.exec():
                custom_dir = os.path.join("models", f"custom_{m.custom_name.value}", f"{cfg.get(cfg.engine)}")

                if os.path.exists(custom_dir):
                    shutil.rmtree(custom_dir)

                os.makedirs(custom_dir, exist_ok=True)
                custom_name = f"custom_{m.custom_name.value}"

                if m.pth_file.value != "Please Select a 'pth' File":
                    shutil.copy(m.pth_file.value, os.path.join(custom_dir, f"{custom_name}_v1.pth"))

                if m.ckpt_file.value != "Please Select a 'ckpt' File":
                    shutil.copy(m.ckpt_file.value, os.path.join(custom_dir, f"{custom_name}_v1.cpkt"))

                if m.index_file.value != "Please Select an 'index' File":
                    shutil.copy(m.index_file.value, os.path.join(custom_dir, f"{custom_name}_v1.index"))

                custom_model = {
                    'name': f"custom_{m.custom_name.value}",
                    'display_name': m.custom_name.value,
                    f"{cfg.get(cfg.engine)}": {
                        "version": "1",
                        "engine_version": "2" if cfg.get(cfg.engine) == 'RVC' or cfg.get(cfg.engine) == 'GPT_SoVITS' else '1',
                        "engine": f"{cfg.get(cfg.engine)}",
                        "type": "pth"
                    }
                }

                if os.path.exists('config/custom_models.json'):
                    with open('config/custom_models.json', 'r', encoding="utf-8") as file:
                        custom_models = json.load(file)
                else:
                    custom_models = []

                custom_models.append(custom_model)

                with open('config/custom_models.json', 'w', encoding="utf-8") as file:
                    json.dump(custom_models, file)

                QTimer.singleShot(0, lambda: (
                    self.parent.load_models_config()
                ))
        else:
            title = 'Disclaimer for Use for Custom Imports'
            content = CUSTOM_DISCLAIMER
            w = MessageBox(title, content, self.parent.window())
            w.yesButton.setText(self.tr('Agree'))
            if w.exec():
                cfg.set(cfg.accepts_custom_disclaimer, True)
                self.add_custom()

    def find_versions(self, directory_path, character_model_name, model_type):
        # Construct the pattern dynamically
        pattern_str = fr"^{character_model_name}.*_v(\d+)\.{model_type}$"
        pattern = re.compile(pattern_str)

        if os.path.exists(directory_path):
            # Get a list of all files in the directory
            all_files = [f for f in os.listdir(directory_path) if os.path.exists(directory_path) and os.path.isfile(os.path.join(directory_path, f))]

            # Extract version numbers from matching files
            versions = set()
            for file_name in all_files:
                match = pattern.match(file_name)
                if match:
                    version = int(match.group(1))
                    versions.add(version)

            return versions
        else:
            return None

    def clear(self):
        self.filter_line_edit.clear()
        self.rvc_checkbox.setChecked(False)
        headers = ["Load", "Name", "Directory", "RVC"]
        model = CharacterTableModel([], headers)
        self.untrained_table.setModel(model)
        headers = ["Load", "Name", "Directory", "Update", "Delete", "RVC"]
        model = CharacterTableModel([], headers)
        self.trained_table.setModel(model)

    def loadTrained(self, parent, trained_characters):
        d = []
        for row, cm in enumerate(trained_characters):
            d.append([f'load', f'{cm["display_name"]}', f'{cm["name"]}', f'rvc', f'delete', f'dl', cm])

        headers = ["Load", "Name", "Directory", "Update", "Delete", "RVC"]
        table_model = CharacterTableModel(d, headers)
        self.trained_table.setModel(table_model)

        for row in range(table_model.rowCount()):
            character_model = table_model.full_data(row, 6)
            if character_model['RVC']:
                widget = QWidget()
                icon_widget = IconWidget()
                icon_widget.setFixedSize(20, 20)
                icon_widget.setIcon(FIF.CHECKBOX)
                rvc_layout = QHBoxLayout(widget)
                rvc_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                rvc_layout.setContentsMargins(1, 1, 1, 1)
                rvc_layout.addWidget(icon_widget)
                index = table_model.index(row, 5)
                self.trained_table.setIndexWidget(index, widget)

            model = character_model[cfg.get(cfg.engine)]
            model_dir = os.path.join("models", character_model["name"], cfg.get(cfg.engine))
            model_files = os.listdir(model_dir) if os.path.isdir(model_dir) else []
            pattern_str = fr"^{character_model['name']}.*_v.*\.{model['type']}$"
            pattern = re.compile(pattern_str)
            downloaded = any(f for f in model_files if pattern.match(f))

            if downloaded:
                delete_button = PushButton('Delete')
                delete_button.setIcon(FallTalkStrokeIcons.DELETE.icon())
                delete_button.setMinimumWidth(115)
                delete_widget = QWidget()
                delete_layout = QHBoxLayout(delete_widget)
                delete_layout.addWidget(delete_button)
                delete_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                delete_layout.setContentsMargins(1, 1, 1, 1)
                delete_button.clicked.connect(
                    lambda _, dn=character_model["display_name"], c=character_model['name'], m=model: parent.delete_model(c, m, dn))
                index = table_model.index(row, 4)
                self.trained_table.setIndexWidget(index, delete_widget)

                load_button = PushButton('Load')
                load_button.setIcon(FIF.SEND)
                load_button.setMinimumWidth(115)
                widget = QWidget()
                layout = QHBoxLayout(widget)
                layout.addWidget(load_button)
                layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.setContentsMargins(1, 1, 1, 1)
                load_button.clicked.connect(
                    lambda _, rvc=character_model['RVC'], c=character_model['name'], r=row, m=model, cm=character_model: parent.load_trained_model(c, cm, rvc))
                index = table_model.index(row, 0)
                self.trained_table.setIndexWidget(index, widget)

                # Check if there is a different version locally
                versions = self.find_versions(os.path.join("models", character_model["name"], cfg.get(cfg.engine)), character_model["name"], model['type'])
                different_version_found = int(model['version']) not in versions
                if not different_version_found and character_model['RVC']:
                    versions = self.find_versions(os.path.join("models", character_model["name"], "RVC"), character_model["name"], character_model['RVC']['type'])
                    different_version_found = versions is None or int(character_model['RVC']['version']) not in versions

                if different_version_found:
                    update_button = PushButton('Update')
                    update_button.setMinimumWidth(115)
                    update_button.setIcon(FIF.UPDATE.icon(color=cfg.get(cfg.themeColor)))

                    widget = QWidget()
                    layout = QHBoxLayout(widget)
                    layout.addWidget(update_button)
                    layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    layout.setContentsMargins(1, 1, 1, 1)
                    update_button.clicked.connect(
                        lambda _, rvc=character_model['RVC'], c=character_model['name'], r=row, m=model: parent.update_model(c, m, rvc))
                    index = table_model.index(row, 3)
                    self.trained_table.setIndexWidget(index, widget)

            else:
                download_button = PushButton('Download')
                download_button.setMinimumWidth(115)
                download_button.setIcon(FIF.CLOUD_DOWNLOAD)

                widget = QWidget()
                layout = QHBoxLayout(widget)
                layout.addWidget(download_button)
                layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.setContentsMargins(1, 1, 1, 1)

                download_button.clicked.connect(
                    lambda _, rvc=character_model['RVC'], c=character_model['name'], r=row, m=model: parent.download_model(c, m, rvc))
                index = table_model.index(row, 0)
                self.trained_table.setIndexWidget(index, widget)

    def loadCustom(self, parent, custom_characters):
        data = []
        for row, cm in custom_characters.items():
            data.append([f'load', cm["display_name"], cm["display_name"] if cm["name"] is None else cm["name"], f'delete', f'rvc', cm])

        headers = ["Load", "Name", "Directory", "Delete", "RVC"]
        table_model = CharacterTableModel(data, headers)
        self.custom_table.setModel(table_model)

        for row in range(table_model.rowCount()):
            character_model = table_model.full_data(row, 5)

            if 'RVC' in character_model:
                widget = QWidget()
                icon_widget = IconWidget()
                icon_widget.setFixedSize(20, 20)
                icon_widget.setIcon(FIF.CHECKBOX)
                rvc_layout = QHBoxLayout(widget)
                rvc_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                rvc_layout.setContentsMargins(1, 1, 1, 1)
                rvc_layout.addWidget(icon_widget)
                index = table_model.index(row, 4)
                self.custom_table.setIndexWidget(index, widget)

            if cfg.get(cfg.engine) in character_model or 'RVC' in character_model:
                load_button = PushButton('Load')
                load_button.setMinimumWidth(115)
                load_button.setIcon(FIF.SEND)

                widget = QWidget()
                layout = QHBoxLayout(widget)
                layout.addWidget(load_button)
                layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.setContentsMargins(1, 1, 1, 1)
                load_button.clicked.connect(lambda _, rvc=character_model['RVC'] if 'RVC' in character_model else None, cm=character_model, c=character_model['name'], r=row: parent.load_custom_model(c, cm, rvc))
                index = table_model.index(row, 0)
                self.custom_table.setIndexWidget(index, widget)

            delete_button = PushButton('Delete')
            delete_button.setIcon(FallTalkStrokeIcons.DELETE.icon())
            delete_button.setMinimumWidth(115)
            delete_widget = QWidget()
            delete_layout = QHBoxLayout(delete_widget)
            delete_layout.addWidget(delete_button)
            delete_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            delete_layout.setContentsMargins(1, 1, 1, 1)
            delete_button.clicked.connect(
                lambda _, dn=character_model["display_name"], c=character_model['name']: parent.delete_custom_model(c, dn))
            index = table_model.index(row, 3)
            self.custom_table.setIndexWidget(index, delete_widget)

    def loadUntrained(self, parent, untrained_characters):
        data = []
        for row, cm in enumerate(untrained_characters):
            data.append([f'load', cm["display_name"], cm["display_name"] if cm["name"] is None else cm["name"], f'rvc', cm])

        headers = ["Load", "Name", "Directory", "RVC"]
        table_model = CharacterTableModel(data, headers)
        self.untrained_table.setModel(table_model)

        for row in range(table_model.rowCount()):
            character_model = table_model.full_data(row, 4)

            if character_model['RVC']:
                widget = QWidget()
                icon_widget = IconWidget()
                icon_widget.setFixedSize(20, 20)
                icon_widget.setIcon(FIF.CHECKBOX)
                rvc_layout = QHBoxLayout(widget)
                rvc_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                rvc_layout.setContentsMargins(1, 1, 1, 1)
                rvc_layout.addWidget(icon_widget)
                index = table_model.index(row, 3)
                self.untrained_table.setIndexWidget(index, widget)

            # Add Load Button for Untrained Characters
            load_button = PushButton('Load')
            load_button.setMinimumWidth(115)
            load_button.setIcon(FIF.SEND)

            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.addWidget(load_button)
            layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.setContentsMargins(1, 1, 1, 1)
            load_button.clicked.connect(lambda _, rvc=character_model['RVC'], c=character_model, r=row: parent.load_base_model(c, rvc))
            index = table_model.index(row, 0)
            self.untrained_table.setIndexWidget(index, widget)

    def on_header_clicked(self, index):
        pass

    def apply_filter(self):
        state = self.rvc_checkbox.isChecked()
        text = self.filter_line_edit.text()

        model = self.untrained_table.model()
        if model:
            for row in range(model.rowCount()):
                match = text.lower() in model.full_data(row, 0).lower() or text.lower() in model.full_data(row, 1).lower()
                if state:
                    rvc_match = model.full_data(row, 4)['RVC'] is not None
                    self.untrained_table.setRowHidden(row, not match or not rvc_match)
                else:
                    self.untrained_table.setRowHidden(row, not match)

        model = self.trained_table.model()
        if model:
            for row in range(model.rowCount()):
                match = text.lower() in model.full_data(row, 0).lower() or text.lower() in model.full_data(row, 1).lower()
                if state:
                    rvc_match = model.full_data(row, 6)['RVC'] is not None
                    self.trained_table.setRowHidden(row, not match or not rvc_match)
                else:
                    self.trained_table.setRowHidden(row, not match)

    def addSubInterface(self, widget: QWidget, objectName, text):
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget)
        )


class AudioGenWidget(FallTalkWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent, text="Sound Generator", vertical=True)
        self.parent = parent

        self.text_input = TextEdit()
        font = QFont()
        font.setPointSize(12)
        self.text_input.setFont(font)
        self.addToFrame(self.text_input)

        self.text_input.setPlaceholderText("""
The first time you generate, a 5GB model must downloaded. It is recommended you have 6GB of VRAM, but it can work on less or CPU mode, just slowly.

dog barking

sirenes of an emergency vehicule

footsteps in a corridor

A baby is crying in a huge room.

Sine wave with low pitch.

Wooden table tapping sound followed by water pouring.

gun reloading

Two swords fighting

Audio model is licensed under CC-By-NC license for non commercial use        
        """)

        self.generate_button = PrimaryPushButton("Generate")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.generate_fx)

        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVolume(100)

        self.output_name = ConfigItem("fx", "output_name", None, ConfigValidator())

        self.autoplay = SwitchSettingCard(
            FIF.PLAY,
            self.tr('Autoplay'),
            self.tr('Automatically Play Generated Audio'),
            cfg.auto_play,
        )
        self.output_name_card = TextSettingCard(
            self.output_name,
            FIF.SAVE_AS,
            self.tr('Output Name'),
            self.tr('Name of Generated WAV file'),
            placeholder="Random"
        )

        self.parse_mode_card = RadioSettingCard(
            cfg.parse_mode,
            FIF.CUT,
            self.tr('Generation Mode'),
            self.tr('Split on commas to generate multiple layers.'),
            texts=["Split on Comma", "Single Command"],
            parent=self
        )

        self.duration_card = SpinSettingCard(
            cfg.fx_duration,
            FIF.STOP_WATCH,
            self.tr('Duration in Seconds'),
            self.tr('Max 120'),
            step=5
        )

        self.temperature_card = RangeSettingCardScaled(
            cfg.music_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Randomness, 1 = balanced, 0 = disabled'),
        )

        self.a_d_ = QGroupBox()
        self.a_d_.setStyleSheet("border: none")
        self.a_d__layout = QHBoxLayout()
        self.a_d__layout.setContentsMargins(0, 0, 0, 0)
        self.a_d__layout.addWidget(self.autoplay, 3)
        self.a_d__layout.addWidget(self.duration_card, 3)
        self.a_d_.setLayout(self.a_d__layout)

        self.r_and_sub = QGroupBox()
        self.r_and_sub.setStyleSheet("border: none")
        self.r_and_sub_layout = QHBoxLayout()
        self.r_and_sub_layout.setContentsMargins(0, 0, 0, 0)
        self.r_and_sub_layout.addWidget(self.output_name_card, 3)
        self.r_and_sub_layout.addWidget(self.temperature_card, 3)
        self.r_and_sub.setLayout(self.r_and_sub_layout)

        self.addToFrame(self.parse_mode_card)
        self.addToFrame(self.r_and_sub)
        self.addToFrame(self.a_d_)
        self.addToFrame(self.generate_button)
        self.addToFrame(self.media_player)


class MusicGenWidget(FallTalkWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent, text="Music Generator", vertical=True)
        self.parent = parent

        self.text_input = TextEdit()
        font = QFont()
        font.setPointSize(12)
        self.text_input.setFont(font)
        self.addToFrame(self.text_input)

        self.text_input.setPlaceholderText("""
The first time you generate, a 10GB model must downloaded. It is recommended you have 12GB of VRAM, but it can work on less or CPU mode, just slowly.

Include some level of details on the instruments present, along with some intended use case (e.g. adding “perfect for a commercial”) can sometimes help. You can control BPM and Time Signatures 4/4, 3/4, 5/4, 2/4, etc, or leave it to the AI. Here are some example prompts:

Violins and synths that inspire awe at the finiteness of life and the universe.

An 80s driving pop song with heavy drums and synth pads in the background

a light and cheerily EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130

3/4 105bpm piano only baroque

Audio model is licensed under CC-By-NC license for non commercial use        
        """)

        self.generate_button = PrimaryPushButton("Generate")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.generate_music)

        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVolume(100)

        self.output_name = ConfigItem("TTS", "output_name", None, ConfigValidator())

        self.autoplay = SwitchSettingCard(
            FIF.PLAY,
            self.tr('Autoplay'),
            self.tr('Automatically Play Generated Audio'),
            cfg.auto_play,
        )

        self.temperature_card = RangeSettingCardScaled(
            cfg.music_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Randomness, 1 = balanced, 0 = disabled'),
        )

        self.extend_stride_card = RangeSettingCard(
            cfg.extend_stride,
            FIF.SKIP_FORWARD,
            self.tr('Extended Strike'),
            self.tr('Higher Number = Faster, Lower = Better Quality'),
        )

        self.output_name_card = TextSettingCard(
            self.output_name,
            FIF.SAVE_AS,
            self.tr('Output Name'),
            self.tr('Name of Generated WAV file'),
            placeholder="Random"
        )

        self.duration_card = SpinSettingCard(
            cfg.music_duration,
            FIF.STOP_WATCH,
            self.tr('Duration in Seconds'),
            self.tr('Max 300'),
            step=5
        )

        self.parse_mode_card = RadioSettingCard(
            cfg.parse_mode,
            FIF.CUT,
            self.tr('Generation Mode'),
            self.tr('Split on commas to generate multiple musical layers.'),
            texts=["Split on Comma", "Single Command"],
            parent=self
        )

        self.mode_card = RadioSettingCard(
            cfg.audio_mode,
            FallTalkStrokeIcons.VOICE_SQUARE.icon(),
            self.tr('Mode'),
            self.tr('Which model should we use? Changing causes loading in next generation'),
            texts=["Mono (5 GB)", "Stereo (12 GB)", "Song (20 GB)"],
            parent=self
        )

        self.ref_file = ConfigItem("audioGen", "ref", "Please Select a File", FileValidator())

        self.ref_file_card = PushSettingCard(
            self.tr('Select File'),
            FIF.DOCUMENT,
            self.tr("Reference Audio to use for Melody"),
            self.ref_file.value,
        )

        self.ref_file_card.clicked.connect(self.__onOutputFolderCardClicked)

        self.f_and_sub = QGroupBox()
        self.f_and_sub.setStyleSheet("border: none")
        self.f_and_sub_layout = QHBoxLayout()
        self.f_and_sub_layout.setContentsMargins(0, 0, 0, 0)
        self.f_and_sub_layout.addWidget(self.output_name_card, 3)
        self.f_and_sub_layout.addWidget(self.ref_file_card, 3)
        self.f_and_sub.setLayout(self.f_and_sub_layout)

        self.r_and_sub = QGroupBox()
        self.r_and_sub.setStyleSheet("border: none")
        self.r_and_sub_layout = QHBoxLayout()
        self.r_and_sub_layout.setContentsMargins(0, 0, 0, 0)
        self.r_and_sub_layout.addWidget(self.autoplay, 3)
        self.r_and_sub_layout.addWidget(self.duration_card, 3)
        self.r_and_sub.setLayout(self.r_and_sub_layout)

        self.t_and_s = QGroupBox()
        self.t_and_s.setStyleSheet("border: none")
        self.t_and_s_layout = QHBoxLayout()
        self.t_and_s_layout.setContentsMargins(0, 0, 0, 0)
        self.t_and_s_layout.addWidget(self.extend_stride_card, 3)
        self.t_and_s_layout.addWidget(self.temperature_card, 3)
        self.t_and_s.setLayout(self.t_and_s_layout)

        self.addToFrame(self.mode_card)
        self.addToFrame(self.parse_mode_card)
        self.addToFrame(self.f_and_sub)
        self.addToFrame(self.r_and_sub)
        self.addToFrame(self.t_and_s)
        self.addToFrame(self.generate_button)
        self.addToFrame(self.media_player)

    def __onOutputFolderCardClicked(self):
        allowed_file_types = "WAV files (*.wav);;MP3 files (*.mp3)"
        folder = QFileDialog.getOpenFileName(
            self, self.tr("Choose CSV or Text File"), "./", allowed_file_types)
        if not folder or folder[0] == "":
            return

        self.ref_file.value = folder[0]
        self.ref_file_card.setContent(folder[0])


class UpscaleWidget(FallTalkWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent, text="Bulk Enhancement", vertical=True)
        self.parent = parent

        self.spacer = QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.boxLayout.addItem(self.spacer)

        self.audio_mode = OptionsConfigItem("upscaler", "mode", "isolate vocals", OptionsValidator(["denoise", "isolate", "upscale"]))

        self.mode_card = RadioSettingCard(
            self.audio_mode,
            FallTalkStrokeIcons.VOICE_SQUARE.icon(),
            self.tr('Mode'),
            self.tr('Upscale 16 kHz or below. Denoise for Recorded Speech. Isolate vocals is an AI denoiser from removing vocals from heavy background noise'),
            texts=["Denoise", "Isolate Vocals", "Upscale"],
            parent=self
        )

        self.sample_rate = OptionsConfigItem("upscaler", "sample", 44100, OptionsValidator([44100, 48000]))

        self.sample_rate_card = RadioSettingCard(
            self.sample_rate,
            FallTalkIcons.SINE.icon(),
            self.tr('Upscaler Sample Rate'),
            self.tr('Fallout 4 Default is 44100Hz'),
            texts=['44100', '48000'],
            parent=self
        )

        self.upscale_dir = ConfigItem("upscale", "upscale_dir", None, CustomFolderValidator())

        self.upscale_dir_card = PushSettingCard(
            self.tr('Select Folder'),
            FIF.FOLDER,
            self.tr("Directory to Enhance"),
            self.upscale_dir.value,
        )

        self.upscale_dir_card.clicked.connect(self.__onFolderCardClicked)

        self.generate_button = PrimaryPushButton("Bulk Enhance")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.upscale_folder)

        self.include_subdir = SwitchSettingCard(
            FIF.FOLDER_ADD,
            self.tr('Sub Directories'),
            self.tr('Include Sub Directories?'),
            cfg.include_subdir,
        )

        self.replace_existing_card = SwitchSettingCard(
            FallTalkIcons.REPLACE.icon(),
            self.tr('Replace'),
            self.tr('Replace all original WAV, XWM, or FUZ'),
            cfg.replace_existing,
        )

        self.mo_sampe = QGroupBox()
        self.mo_sampe.setStyleSheet("border: none")
        self.mo_sampe_layout = QHBoxLayout()
        self.mo_sampe_layout.setContentsMargins(0, 0, 0, 0)
        self.mo_sampe_layout.addWidget(self.upscale_dir_card, 3)
        self.mo_sampe_layout.addWidget(self.sample_rate_card, 3)
        self.mo_sampe.setLayout(self.mo_sampe_layout)

        self.r_and_sub = QGroupBox()
        self.r_and_sub.setStyleSheet("border: none")
        self.r_and_sub_layout = QHBoxLayout()
        self.r_and_sub_layout.setContentsMargins(0, 0, 0, 0)
        self.r_and_sub_layout.addWidget(self.include_subdir, 3)
        self.r_and_sub_layout.addWidget(self.replace_existing_card, 3)
        self.r_and_sub.setLayout(self.r_and_sub_layout)

        self.addToFrame(self.mode_card)
        self.addToFrame(self.mo_sampe)
        self.addToFrame(self.r_and_sub)
        self.addToFrame(self.generate_button)

    def __onFolderCardClicked(self):
        """ download folder card clicked slot """
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Choose A Directory"), "./")
        if not folder or folder == "":
            return

        self.upscale_dir.value = folder
        self.upscale_dir_card.setContent(folder)


class BulkLipFuzWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.lip_dir = ConfigItem("bulk", "lip_dir", None, CustomFolderValidator())

        self.threads_card = SpinSettingCard(
            cfg.threads,
            FIF.STOP_WATCH,
            self.tr('Threads for processing'),
            self.tr('Experimental: Monitor GPU or CPU.'),
            step=1
        )

        self.lip_dir_card = PushSettingCard(
            self.tr('Select Folder'),
            FIF.FOLDER,
            self.tr("Directory for bulk LIP / FIZ"),
            self.lip_dir.value,
        )

        self.include_subdir = SwitchSettingCard(
            FIF.FOLDER_ADD,
            self.tr('Sub Directories'),
            self.tr('Include Sub Directories?'),
            cfg.include_subdir,
        )

        # self.replace_existing_card = SwitchSettingCard(
        #     FallTalkIcons.REPLACE.icon(),
        #     self.tr('Replace'),
        #     self.tr('Replace all original WAV, XWM'),
        #     cfg.replace_existing,
        # )

        self.delete_leftovers = SwitchSettingCard(
            FIF.DELETE,
            self.tr('Keep Only FUZ'),
            self.tr('Delete XMW, LIP, and WAV'),
            cfg.keep_only_fuz
        )

        self.gen_settings = QGroupBox()
        self.gen_settings.setStyleSheet("border: none")
        self.gen_settings_layout = QHBoxLayout(self.gen_settings)
        self.gen_settings_layout.setContentsMargins(0, 0, 0, 0)
        self.gen_settings_layout.addWidget(self.lip_dir_card, 2)
        self.gen_settings.setLayout(self.gen_settings_layout)

        self.f_c_ = QGroupBox()
        self.f_c_.setStyleSheet("border: none")
        self.f_c__layout = QHBoxLayout(self.f_c_)
        self.f_c__layout.setContentsMargins(0, 0, 0, 0)
        self.f_c__layout.addWidget(self.include_subdir, 3)
        self.f_c__layout.addWidget(self.delete_leftovers, 3)
        self.f_c_.setLayout(self.f_c__layout)

        self.setContentsMargins(0, 0, 0, 0)

        self.spacer = QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.fuz_widget_view = QVBoxLayout(self)
        self.fuz_widget_view.setContentsMargins(0, 0, 0, 0)
        self.fuz_widget_view.addItem(self.spacer)
        self.fuz_widget_view.addWidget(self.threads_card)
        self.fuz_widget_view.addWidget(self.gen_settings)
        self.fuz_widget_view.addWidget(self.f_c_)
        # self.rvc_widget_view.addWidget(self.r_and_sub)

        self.lip_dir_card.clicked.connect(self.__onFolderCardClicked)

    def __onFolderCardClicked(self):
        """ download folder card clicked slot """
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Choose A Directory"), "./")
        if not folder or folder == "":
            return

        self.lip_dir.value = folder
        self.lip_dir_card.setContent(folder)


class BulkGenerationRVCWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rvc_dir = ConfigItem("bulk", "rvc_dir", None, CustomFolderValidator())

        self.threads_card = SpinSettingCard(
            cfg.threads,
            FIF.STOP_WATCH,
            self.tr('Threads for processing'),
            self.tr('Experimental: Monitor GPU and CPU.'),
            step=1
        )

        self.rvc_dir_card = PushSettingCard(
            self.tr('Select Folder'),
            FIF.FOLDER,
            self.tr("Directory for bulk RVC"),
            self.rvc_dir.value,
        )

        self.rvc_dir_card.clicked.connect(self.__onFolderCardClicked)

        self.include_subdir = SwitchSettingCard(
            FIF.FOLDER_ADD,
            self.tr('Sub Directories'),
            self.tr('Include Sub Directories?'),
            cfg.include_subdir,
        )

        self.replace_existing_card = SwitchSettingCard(
            FallTalkIcons.REPLACE.icon(),
            self.tr('Replace'),
            self.tr('Replace all original WAV, XWM, or FUZ'),
            cfg.replace_existing,
        )

        self.r_and_sub = QGroupBox()
        self.r_and_sub.setStyleSheet("border: none")
        self.r_and_sub_layout = QHBoxLayout(self.r_and_sub)
        self.r_and_sub_layout.setContentsMargins(0, 0, 0, 0)
        self.r_and_sub_layout.addWidget(self.include_subdir, 3)
        self.r_and_sub_layout.addWidget(self.replace_existing_card, 3)
        self.r_and_sub.setLayout(self.r_and_sub_layout)

        self.character_card = RvcComboBoxSettingsCard(
            FIF.PEOPLE,
            self.tr('Character'),
            self.tr('Which Character to Use'))

        self.xwm_card = SwitchSettingCard(
            FIF.COMMAND_PROMPT,
            self.tr('Create FUZ'),
            self.tr('Create XWM, LIP, and FUZ'),
            cfg.xwm_enabled,
        )

        self.delete_leftovers = SwitchSettingCard(
            FIF.DELETE,
            self.tr('Keep Only FUZ'),
            self.tr('Delete XMW, LIP, and WAV'),
            cfg.keep_only_fuz
        )

        self.use_existing_lip = SwitchSettingCard(
            FIF.SHARE,
            self.tr('Use Existing LIP'),
            self.tr('Use existing LIP if it exists or generate new'),
            cfg.use_existing_lip
        )

        self.gen_settings = QGroupBox()
        self.gen_settings.setStyleSheet("border: none")
        self.gen_settings_layout = QHBoxLayout(self.gen_settings)
        self.gen_settings_layout.setContentsMargins(0, 0, 0, 0)
        self.gen_settings_layout.addWidget(self.rvc_dir_card, 2)
        self.gen_settings_layout.addWidget(self.character_card, 2)
        self.gen_settings.setLayout(self.gen_settings_layout)

        self.f_c_ = QGroupBox()
        self.f_c_.setStyleSheet("border: none")
        self.f_c__layout = QHBoxLayout(self.f_c_)
        self.f_c__layout.setContentsMargins(0, 0, 0, 0)
        self.f_c__layout.addWidget(self.xwm_card, 3)
        self.f_c__layout.addWidget(self.delete_leftovers, 3)
        self.f_c__layout.addWidget(self.use_existing_lip, 3)
        self.f_c_.setLayout(self.f_c__layout)

        self.rvc_protect_card = RangeSettingCardScaled(
            cfg.rvc_protect,
            FIF.BROOM,
            self.tr("Breath Sounds Envelope"),
            self.tr("Prevents sound artifacts and breath noises"),
        )

        self.rvc_filter_radius_card = RangeSettingCard(
            cfg.rvc_filter_radius,
            FIF.FILTER,
            self.tr("Filter Radius"),
            self.tr("If >= 3, potential to decrease respiration."),
        )

        self.p_f = QGroupBox()
        self.p_f.setStyleSheet("border: none")
        self.p_f_layout = QHBoxLayout(self.p_f)
        self.p_f_layout.setContentsMargins(0, 0, 0, 0)
        self.p_f_layout.addWidget(self.rvc_protect_card, 3)
        self.p_f_layout.addWidget(self.rvc_filter_radius_card, 3)
        self.p_f.setLayout(self.p_f_layout)

        self.rvc_pitch_card = RangeSettingCard(
            cfg.rvc_pitch,
            FIF.MARKET,
            self.tr("Pitch Adjustment"),
            self.tr("Set the pitch of the audio, for opposite gender."),
        )

        self.p_t = QGroupBox()
        self.p_t.setStyleSheet("border: none")
        self.p_t_layout = QHBoxLayout(self.p_t)
        self.p_t_layout.setContentsMargins(0, 0, 0, 0)
        self.p_t_layout.addWidget(self.threads_card, 3)
        self.p_t_layout.addWidget(self.rvc_pitch_card, 3)
        self.p_t.setLayout(self.p_t_layout)

        self.setContentsMargins(0, 0, 0, 0)

        self.spacer = QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.rvc_widget_view = QVBoxLayout(self)
        self.rvc_widget_view.setContentsMargins(0, 0, 0, 0)
        self.rvc_widget_view.addItem(self.spacer)
        self.rvc_widget_view.addWidget(self.gen_settings)
        self.rvc_widget_view.addWidget(self.p_t)
        self.rvc_widget_view.addWidget(self.p_f)
        self.rvc_widget_view.addWidget(self.r_and_sub)
        self.rvc_widget_view.addWidget(self.f_c_)

    def __onFolderCardClicked(self):
        """ download folder card clicked slot """
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Choose A Directory"), "./")
        if not folder or folder == "":
            return

        self.rvc_dir.value = folder
        self.rvc_dir_card.setContent(folder)


class BulkGenerationTableWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bulk_table = TableView()
        self.bulk_table.setBorderVisible(True)
        self.bulk_table.setBorderRadius(8)
        self.bulk_table.setAlternatingRowColors(True)
        self.bulk_table.setWordWrap(False)
        self.bulk_table.verticalHeader().setVisible(False)
        self.bulk_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)

        self.upload_file = ConfigItem("bulk", "upload_file", "Please Select a File", FileValidator())

        self.upload_file_card = PushSettingCard(
            self.tr('Select File'),
            FIF.DOCUMENT,
            self.tr("CSV or TXT file matching the table"),
            self.upload_file.value,
        )

        self.started_card = PushSettingCard(
            self.tr('Getting Started'),
            FIF.QUESTION,
            self.tr("Guide"),
            self.tr("Required Fields and Schema"),
        )

        self.f_and_u = QGroupBox()
        self.f_and_u.setStyleSheet("border: none")
        self.f_and_u_layout = QHBoxLayout()
        self.f_and_u_layout.setContentsMargins(0, 0, 0, 0)
        self.f_and_u_layout.addWidget(self.upload_file_card, 1)
        self.f_and_u_layout.addWidget(self.started_card, 1)
        self.f_and_u.setLayout(self.f_and_u_layout)

        self.headers = ["filename", "character", "text", "reference", "output_dir"]
        model = TableModel([], self.headers)
        self.bulk_table.setModel(model)
        self.bulk_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.bulk_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.bulk_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.bulk_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.bulk_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.setContentsMargins(0, 0, 0, 0)

        self.bulk_widget_view = QVBoxLayout(self)
        self.bulk_widget_view.setContentsMargins(0, 0, 0, 0)

        self.bulk_widget_view.addWidget(self.bulk_table)
        self.bulk_widget_view.addWidget(self.f_and_u)

        self.xwm_card = SwitchSettingCard(
            FIF.COMMAND_PROMPT,
            self.tr('Create FUZ'),
            self.tr('Create XWM, LIP, and FUZ'),
            cfg.xwm_enabled,
        )
        self.delete_leftovers = SwitchSettingCard(
            FIF.DELETE,
            self.tr('Keep Only FUZ'),
            self.tr('Delete XMW, LIP, and WAV'),
            cfg.keep_only_fuz
        )
        self.rvc_enabled = SwitchSettingCard(
            FIF.MEGAPHONE,
            self.tr('RVC'),
            self.tr('Use RVC Upscaler (Recommended)'),
            cfg.rvc_enabled
        )
        self.gen_settings = QGroupBox()
        self.gen_settings.setStyleSheet("border: none")
        self.gen_settings_layout = QHBoxLayout(self.gen_settings)
        self.gen_settings_layout.setContentsMargins(0, 0, 0, 0)

        self.gen_settings_layout.addWidget(self.xwm_card, 2)
        self.gen_settings_layout.addWidget(self.delete_leftovers, 2)
        self.gen_settings_layout.addWidget(self.rvc_enabled, 2)
        self.gen_settings.setLayout(self.gen_settings_layout)

        self.bulk_widget_view.addWidget(self.gen_settings)


class BulkGenerationWidget(FallTalkWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent, text="Bulk Generation", vertical=True)
        self.parent = parent

        self.pivot = SegmentedWidget(self)
        self.stackedWidget = QStackedWidget(self)

        # Create a TabView instance
        self.bulk_csv_widget = BulkGenerationTableWidget(self)
        self.bulk_rvc_widget = BulkGenerationRVCWidget(self)
        self.bulk_fuz_widget = BulkLipFuzWidget(self)

        self.bulk_csv_widget.upload_file_card.clicked.connect(self.__onOutputFolderCardClicked)
        self.bulk_csv_widget.started_card.clicked.connect(self.__onShowGettingStarted)

        self.addSubInterface(self.bulk_csv_widget, 'bulk_csv_widget', 'CSV')
        self.addSubInterface(self.bulk_rvc_widget, 'bulk_rvc_widget', 'RVC')
        self.addSubInterface(self.bulk_fuz_widget, 'bulk_fuz_widget', 'FUZ')

        self.generate_button = PrimaryPushButton("Bulk Generate Audio")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.bulk_inference)

        self.boxLayout.addWidget(self.pivot, 0, Qt.AlignmentFlag.AlignLeft)
        self.boxLayout.addWidget(self.stackedWidget)
        self.stackedWidget.currentChanged.connect(self.onCurrentIndexChanged)
        self.stackedWidget.setCurrentWidget(self.bulk_csv_widget)
        self.pivot.setCurrentItem(self.bulk_csv_widget.objectName())

        self.addToFrame(self.generate_button)

        self.setEnabled(cfg.engine.value != 'VoiceCraft')

    def onCurrentIndexChanged(self, index):
        if self.stackedWidget.currentWidget() == self.bulk_rvc_widget:
            if self.bulk_rvc_widget.character_card.configItem.size() == 0:
                self.populate_character_card()

        widget = self.stackedWidget.widget(index)
        self.pivot.setCurrentItem(widget.objectName())

    def addSubInterface(self, widget: QWidget, objectName, text):
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget)
        )

    def populate_character_card(self):
        self.bulk_rvc_widget.character_card.configItem.clear()
        items = []
        for key, value in self.parent.models.items():
            if 'RVC' in value:
                items.append(ComboItem(value['display_name'], userData=value))

        if self.parent.custom_models is not None:
            for key, value in self.parent.custom_models.items():
                if 'RVC' in value:
                    items.append(ComboItem(value['display_name'], userData=value))

        for i in sorted(items, key=lambda x: x.text):
            self.bulk_rvc_widget.character_card.configItem.addItem(i.text, userData=i.userData)

        self.bulk_rvc_widget.character_card.configItem.setCurrentIndex(0)

    def __onShowGettingStarted(self):

        title = 'Getting Started Guide'
        content = """
        The bulk file generation will use the currently loaded TTS engine, except voicecraft as its not suitable. 
        
        Recommendations are StyleTTS2 or GPT_SoVITS. RVC will run with each engine as needed, you do not need to set the engine to RVC. 
        
        Accepts a CSV file with no header. Fields must be int he following order:
        
        "filename",  "character", "text", "reference", output_dir
        
        fileName <optional>: The name of the output, will be randomly generated if blank
            - accepted values: a0231s_1, a0231s_2.wav, or blank
            
        character <required>: The fallout4 game name of the character. 
            - accepted values: playervoicemale01, robotsentrybot, etc
            
        text: <required>: The text you would like to regenerate.
            - tts accepted value: text string,
            - rvc accepted value: path of a file example: samples/sentrybot_falltalk_rvc.wav, C:/Audio/Sample.wav
            
        reference <required>: You are using a "TTS" engine. Not needed for RVC.
            - tts accepted value: fuz file name: 00091381_1.fuz, 
            
        output_dir <optional>: 
            - accepted values: any valid directory C:\\Output, ./output, etc, 
            
        Each bulk run will be placed in the bulk_outputs folder in the main directory. 
        
        Once data is loaded in the table, you can edit each field as needed.
        
        """

        w = MessageBox(title, content, self.parent.window())
        if w.exec():
            pass

    def __onOutputFolderCardClicked(self):

        allowed_file_types = "Text files (*.txt);;CSV files (*.csv)"
        folder = QFileDialog.getOpenFileName(
            self, self.tr("Choose CSV or Text File"), "./", allowed_file_types)
        if not folder or folder[0] == "":
            return

        self.clear()
        self.bulk_csv_widget.upload_file.value = folder[0]
        self.bulk_csv_widget.upload_file_card.setContent(folder[0])

        data = []
        with open(self.bulk_csv_widget.upload_file.value, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file, self.bulk_csv_widget.headers)
            for row in csv_reader:
                data.append([row['filename'], row['character'], row['text'], row['reference'], row['output_dir']])

        model = TableModel(data, self.bulk_csv_widget.headers)
        self.bulk_csv_widget.bulk_table.setModel(model)

    def clear(self):
        model = TableModel([], self.bulk_csv_widget.headers)
        self.bulk_csv_widget.bulk_table.setModel(model)


class CustomTableModel(QAbstractTableModel):
    def __init__(self, data, headers, parent=None):
        super().__init__(parent)
        self._data = sorted(data, key=lambda x: len(x[1]), reverse=True)
        self._headers = headers
        self._selected_rows = set()

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self._data[index.row()][index.column()]
        elif role == Qt.ItemDataRole.BackgroundRole:
            if index.row() in self._selected_rows:
                color = QColor(cfg.get(cfg.themeColor))
                color.setAlpha(128)
                return color
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._headers[section]
            if orientation == Qt.Orientation.Vertical:
                return section + 1
        return None

    def toggle_selection(self, row):
        if row in self._selected_rows:
            self._selected_rows.remove(row)
        else:
            self._selected_rows.add(row)
        self.redraw(row)

    def redraw(self, row):
        self.dataChanged.emit(self.index(row, 0), self.index(row, len(self._headers) - 1))


class CustomReferencesModel(QAbstractTableModel):
    def __init__(self, data, headers, parent=None):
        super().__init__(parent)
        self._data = sorted(data, key=lambda x: len(x[0]), reverse=True)
        self._headers = headers
        self._selected_rows = set()

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self._data[index.row()][index.column()]
        elif role == Qt.ItemDataRole.BackgroundRole:
            if index.row() in self._selected_rows:
                color = QColor(cfg.get(cfg.themeColor))
                color.setAlpha(128)
                return color
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._headers[section]
            if orientation == Qt.Orientation.Vertical:
                return section + 1
        return None

    def toggle_selection(self, row):
        if row in self._selected_rows:
            self._selected_rows.remove(row)
        else:
            self._selected_rows.add(row)
        self.redraw(row)

    def redraw(self, row):
        self.dataChanged.emit(self.index(row, 0), self.index(row, len(self._headers) - 1))


class ReferencesWidget(FallTalkWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent, text="Reference Audio", vertical=True)
        self.parent = parent
        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVolume(100)

        self.pivot = SegmentedWidget(self)
        self.stackedWidget = QStackedWidget(self)

        self.reference_audio = []
        self.reference_audio_length = 0.0

        self.reference_table = TableView()
        self.reference_table.setBorderRadius(8)
        self.reference_table.setAlternatingRowColors(True)
        self.reference_table.setWordWrap(False)
        self.reference_table.horizontalHeader().setVisible(True)
        self.reference_table.verticalHeader().setVisible(False)
        self.reference_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.reference_table.horizontalHeader().setStretchLastSection(True)
        self.reference_table.setMinimumSize(500, 300)
        self.reference_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.reference_table.setSortingEnabled(True)
        self.reference_table.clicked.connect(self.on_reference_select)
        self.reference_table.setBorderVisible(True)

        headers = ['filename', 'dialogue', 'arcname', 'plugin', 'folder']
        model = CustomTableModel([], headers)
        self.reference_table.setModel(model)
        self.reference_table.setColumnHidden(2, True)
        self.reference_table.setColumnHidden(3, True)
        self.reference_table.setColumnHidden(4, True)

        self.reference_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.reference_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.reference_table.setColumnWidth(0, 150)

        self.controlsBox = QHBoxLayout()
        self.filter_line_edit = SearchLineEdit()
        self.filter_line_edit.setPlaceholderText("Filter...")
        self.filter_line_edit.textChanged.connect(self.apply_filter)
        self.controlsBox.addWidget(self.filter_line_edit, stretch=5)

        self.select_button = PushButton("Select")
        self.select_button.clicked.connect(self.select_row)
        self.select_button.setIcon(FIF.ADD_TO)
        self.remove_button = PushButton("Remove")
        self.remove_button.setIcon(FIF.REMOVE_FROM)
        self.remove_button.clicked.connect(self.remove_row)
        self.highlight_checkbox = CheckBox("Show Highlighted")
        self.highlight_checkbox.setMinimumWidth(120)
        self.highlight_checkbox.stateChanged.connect(self.apply_filter)
        self.controlsBox.addWidget(self.highlight_checkbox)
        self.controlsBox.addWidget(self.select_button, stretch=2)
        self.controlsBox.addWidget(self.remove_button, stretch=2)
        self.controlsBox.stretch(1)

        self.custom_reference_table = TableView()
        self.custom_reference_table.setBorderRadius(8)
        self.custom_reference_table.setAlternatingRowColors(True)
        self.custom_reference_table.setWordWrap(False)
        self.custom_reference_table.horizontalHeader().setVisible(True)
        self.custom_reference_table.verticalHeader().setVisible(False)
        self.custom_reference_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.custom_reference_table.horizontalHeader().setStretchLastSection(True)
        self.custom_reference_table.setMinimumSize(500, 300)
        self.custom_reference_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.custom_reference_table.setSortingEnabled(True)
        self.custom_reference_table.clicked.connect(self.on_custom_reference_select)
        self.custom_reference_table.setBorderVisible(True)

        self.addSubInterface(self.reference_table, 'reference_table', 'Fallout 4')
        self.addSubInterface(self.custom_reference_table, 'custom_reference_table', 'Custom')

        self.boxLayout.addWidget(self.pivot, 0, Qt.AlignmentFlag.AlignLeft)
        self.boxLayout.addWidget(self.stackedWidget)
        self.boxLayout.addLayout(self.controlsBox)

        self.stackedWidget.currentChanged.connect(self.onCurrentIndexChanged)
        self.stackedWidget.setCurrentWidget(self.reference_table)
        self.pivot.setCurrentItem(self.reference_table.objectName())

        self.load_files_from_folder("references")
        self.addToFrame(self.media_player)

    def addDataToReferencesTable(self, selected_model):
        self.clear()
        data = []
        name = selected_model["name"]
        if not name.startswith("custom_"):
            for audio in selected_model["voicefiles"]:
                if '_custom' in name:
                    pattern = r'_custom\d*$'
                    name = re.sub(pattern, '', name)

                data.append([audio['filename'], audio['dialogue'], audio['arcname'], audio['plugin'], name])
        headers = ['filename', 'dialogue', 'arcname', 'plugin', 'folder']
        model = CustomTableModel(data, headers)
        self.reference_table.setModel(model)

    def apply_filter(self):
        text = self.filter_line_edit.text()
        state = self.highlight_checkbox.isChecked()

        model = self.reference_table.model()
        if model:
            for row in range(model.rowCount()):
                match = any(text.lower() in model.data(model.index(row, col)).lower() for col in range(model.columnCount()))
                if state:
                    self.reference_table.setRowHidden(row, not match or row not in model._selected_rows)

                else:
                    self.reference_table.setRowHidden(row, not match)

        model = self.custom_reference_table.model()
        if model:
            for row in range(model.rowCount()):
                match = any(text.lower() in model.data(model.index(row, col)).lower() for col in range(model.columnCount()))
                if state:
                    self.custom_reference_table.setRowHidden(row, not match or row not in model._selected_rows)
                else:
                    self.custom_reference_table.setRowHidden(row, not match)

    def addSubInterface(self, widget: QWidget, objectName, text):
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget)
        )

    def load_files_from_folder(self, folder_path):
        self.parent.voicecraft_widget.clear(),
        self.parent.gpt_sovits_widget.clear(),

        os.makedirs(folder_path, exist_ok=True)
        files = os.listdir(folder_path)
        data = []
        for row, file_name in enumerate(files):
            data.append([file_name])
        headers = ['filename']
        model = CustomReferencesModel(data, headers)
        self.custom_reference_table.setModel(model)

    def onCurrentIndexChanged(self, index):
        widget = self.stackedWidget.widget(index)
        self.pivot.setCurrentItem(widget.objectName())

    def on_custom_reference_select(self, index):
        if index.isValid():
            player = self.media_player.player
            self.media_player.player.stop()
            row = index.row()
            model = self.custom_reference_table.model()
            item = {}
            for col in range(model.columnCount()):
                key = model.headerData(col, Qt.Orientation.Horizontal)
                value = model.data(model.index(row, col))
                item[key] = value

            QTimer.singleShot(0, lambda: (
                player.setSource(QUrl.fromLocalFile(f"references/{item['filename']}")),
            ))

    def on_reference_select(self, index):
        if index.isValid() and cfg.get(cfg.fallout_4_directory) != "fallout4.exe not found":
            player = self.media_player.player
            self.media_player.player.stop()

            row = index.row()
            model = self.reference_table.model()
            item = {}
            for col in range(model.columnCount()):
                key = model.headerData(col, Qt.Orientation.Horizontal)
                value = model.data(model.index(row, col))
                item[key] = value

            filename = item['filename'].rsplit('.', 1)[0]
            if os.path.exists(f"temp/{filename}.wav"):
                QTimer.singleShot(0, lambda: (
                    player.setSource(QUrl.fromLocalFile(f"temp/{filename}.wav"))
                ))
            else:
                QTimer.singleShot(0, lambda: (
                    falltalkutils.extract_bsa(item),
                    falltalkutils.extract_fuz(os.path.abspath(f"temp/{filename}.fuz")),
                    falltalkutils.create_xwm(os.path.abspath(f"temp/{filename}.xwm"), os.path.abspath(f"temp/{filename}.wav"), False),

                    os.path.exists(f"temp/{filename}.xwm") and os.remove(f"temp/{filename}.xwm"),
                    os.path.exists(f"temp/{filename}.fuz") and os.remove(f"temp/{filename}.fuz"),
                    os.path.exists(f"temp/{filename}.lip") and os.remove(f"temp/{filename}.lip"),

                    player.setSource(QUrl.fromLocalFile(f"temp/{filename}.wav"))
                ))

    def clear(self):
        self.reference_table.clearSelection()
        self.custom_reference_table.clearSelection()
        self.reference_audio = []
        self.reference_audio_length = 0
        # self.settingLabel.setText(self.tr(self.title))
        model = self.custom_reference_table.model()
        if model:
            model._selected_rows = set()
            for row in range(model.rowCount()):
                model.redraw(row)

        model = self.reference_table.model()
        if model:
            model._selected_rows = set()
            for row in range(model.rowCount()):
                model.redraw(row)

        self.highlight_checkbox.setCheckState(Qt.CheckState.Unchecked)

    def select_row(self):
        index = self.stackedWidget.currentWidget().currentIndex()
        if index.isValid():
            self.parent.voicecraft_widget.onReferenceSelect()
            self.parent.gpt_sovits_widget.onReferenceSelect()
            row = index.row()
            model = self.stackedWidget.currentWidget().model()
            if (self.stackedWidget.currentWidget() == self.custom_reference_table):
                file_path = f"references/{model.data(model.index(row, 0))}"
                if file_path not in self.reference_audio:
                    self.reference_audio.append(file_path)
                    self.increase(file_path)
                    model.toggle_selection(row)
            else:
                filename = model.data(model.index(row, 0)).rsplit('.', 1)[0]
                file_path = f"temp/{filename}.wav"
                if file_path not in self.reference_audio:
                    self.reference_audio.append(file_path)
                    self.increase(file_path)
                    model.toggle_selection(row)

    def increase(self, file_path):
        data, samplerate = sf.read(file_path)
        length_in_seconds = len(data) / samplerate
        self.reference_audio_length += length_in_seconds
        self.parent.reference_time_label.setText(self.tr(self._formatTime(self.reference_audio_length)))

    def remove_row(self):
        index = self.stackedWidget.currentWidget().currentIndex()
        if index.isValid():
            self.parent.voicecraft_widget.onReferenceSelect()
            self.parent.gpt_sovits_widget.onReferenceSelect()
            row = index.row()
            model = self.stackedWidget.currentWidget().model()
            if (self.stackedWidget.currentWidget() == self.custom_reference_table):
                file_path = f"references/{model.data(model.index(row, 0))}"
                if file_path in self.reference_audio:
                    self.reference_audio.remove(file_path)
                    model.toggle_selection(row)
                    self.decrease(file_path)

            else:
                filename = model.data(model.index(row, 0)).rsplit('.', 1)[0]
                file_path = f"temp/{filename}.wav"
                if file_path in self.reference_audio:
                    self.reference_audio.remove(file_path)
                    model.toggle_selection(row)
                    self.decrease(file_path)

    def decrease(self, file_path):
        data, samplerate = sf.read(file_path)
        length_in_seconds = len(data) / samplerate
        self.reference_audio_length -= length_in_seconds
        self.parent.reference_time_label.setText(self.tr(self._formatTime(self.reference_audio_length)))

    def _formatTime(self, time: float):
        s = int(time)
        ms = int((time - s) * 1000)
        ms_s = str(ms).zfill(2)[:2]
        return f'{s:02}:{ms_s}'
