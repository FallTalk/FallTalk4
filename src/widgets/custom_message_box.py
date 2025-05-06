import os

from PySide6.QtGui import QFont, Qt
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QFileDialog, QLabel
from qfluentwidgets import FluentIcon as FIF, RangeSettingCard, TextEdit, PrimaryPushButton, ConfigItem, SwitchSettingCard, ConfigValidator, PushSettingCard, MessageBoxBase

from audio.audio_player import StandardAudioPlayerBar
from src.config.config import cfg, FileValidator
from src.utils.icons import FallTalkIcons
from src.ui.cards import RangeSettingCardScaled, RadioSettingCard, TextSettingCard, SpinSettingCard, ComboBoxWordsCard
from src.widgets import GenerationWidget

from src.utils.logging_utils import logger


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
