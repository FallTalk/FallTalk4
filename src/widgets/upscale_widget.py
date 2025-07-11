from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

from PySide6 import QtWidgets
from PySide6.QtWidgets import QHBoxLayout, QGroupBox, QFileDialog, QSpacerItem, QWidget
from qfluentwidgets import FluentIcon as FIF, PrimaryPushButton, SwitchSettingCard, ConfigItem, PushSettingCard, \
    OptionsValidator, OptionsConfigItem, ToolButton

from src.config.config import cfg, CustomFolderValidator
from src.ui.cards import RadioSettingCard
from src.utils.icons import FallTalkIcons
from src.widgets.falltalk_widget import FallTalkWidget
from src.widgets.drawer import RightDrawer
from src.help.upscale_help import UpscaleHelp

class UpscaleWidget(FallTalkWidget):
    def __init__(self, parent: FallTalkApp):
        super().__init__(parent=parent, text="Bulk Enhancement", vertical=True)
        self.parent = parent

        self.spacer = QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.boxLayout.addItem(self.spacer)

        self.audio_mode = OptionsConfigItem("upscaler", "mode", "isolate vocals", OptionsValidator(["denoise", "isolate", "upscale"]))

        self.mode_card = RadioSettingCard(
            self.audio_mode,
            FallTalkIcons.VOICE_SQUARE.icon(stroke=True),
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

        self.generate_button = PrimaryPushButton(text="Bulk Enhance")
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

        # Buttons layout
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)

        self.help_drawer = RightDrawer(self, title="About", icon=FIF.QUESTION)
        self.settings_drawer = RightDrawer(self, title="Advanced Settings", icon=FIF.SETTING)
        self.help_drawer.addWidget(UpscaleHelp(self))

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

        self.buttons_layout.addWidget(self.generate_button)
        self.buttons_layout.addWidget(self.help_button)
        # self.buttons_layout.addWidget(self.settings_button)

        # Create a widget to hold the buttons layout
        self.buttons_widget = QWidget()
        self.buttons_widget.setLayout(self.buttons_layout)


        self.addToFrame(self.buttons_widget)

    def __onFolderCardClicked(self):
        """ download folder card clicked slot """
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Choose A Directory"), "./")
        if not folder or folder == "":
            return

        self.upscale_dir.value = folder
        self.upscale_dir_card.setContent(folder)

    def toggle_settings_drawer(self):
        self.settings_drawer.open_drawer()

    def toggle_help_drawer(self):
        self.help_drawer.open_drawer()