from __future__ import annotations
from typing import TYPE_CHECKING

from PySide6.QtGui import QFont


if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp
    
import csv

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QGroupBox, QHeaderView, QAbstractItemView, QFileDialog, QSpacerItem
from qfluentwidgets import (
    FluentIcon as FIF, TableView, SegmentedWidget,
    SwitchSettingCard, ConfigItem, PushSettingCard, RangeSettingCard, PrimaryPushButton, MessageBox, ToolButton,
    BodyLabel, TextEdit, StrongBodyLabel
)
from qfluentwidgets.components.widgets.combo_box import ComboItem

from src.config.config import cfg, CustomFolderValidator, FileValidator
from src.ui.cards import SpinSettingCard, RvcComboBoxSettingsCard, RangeSettingCardScaled
from src.utils.icons import FallTalkIcons
from src.widgets.falltalk_widget import FallTalkWidget
from src.widgets.table_models import TableModel
from src.widgets.drawer import RightDrawer
from src.settings.rvc_settings import RVCSettings
from src.help.bulk_csv_help import BulkCSVHelp
from src.help.bulk_fuz_help import BulkFuzHelp
from src.help.bulk_rvc_help import BulkRVCHelp


class BaseBulkWidget(QWidget):

    def __init__(self, parent: FallTalkApp):
        super().__init__(parent)
        self.parent = parent
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

        self.generate_button = PrimaryPushButton(text="Bulk Generate Audio")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.bulk_inference)

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(self.generate_button, stretch=1)


    def toggle_settings_drawer(self):
        self.settings_drawer.open_drawer()

    def toggle_help_drawer(self):
        self.help_drawer.open_drawer()

class BulkLipFuzWidget(BaseBulkWidget):
    def __init__(self, parent: FallTalkApp):
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
        self.help_drawer.addWidget(BulkFuzHelp(self))

        self.buttons_layout.addWidget(self.help_button)
        self.fuz_widget_view.addLayout(self.buttons_layout)
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


class BulkGenerationRVCWidget(BaseBulkWidget):
    def __init__(self, parent: FallTalkApp):
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
        self.gen_settings_layout.addWidget(self.rvc_dir_card, 3)
        self.gen_settings_layout.addWidget(self.character_card, 3)
        self.gen_settings_layout.addWidget(self.threads_card, 3)

        self.gen_settings.setLayout(self.gen_settings_layout)

        self.f_c_ = QGroupBox()
        self.f_c_.setStyleSheet("border: none")
        self.f_c__layout = QHBoxLayout(self.f_c_)
        self.f_c__layout.setContentsMargins(0, 0, 0, 0)
        self.f_c__layout.addWidget(self.xwm_card, 3)
        self.f_c__layout.addWidget(self.delete_leftovers, 3)
        self.f_c__layout.addWidget(self.use_existing_lip, 3)

        self.f_c_.setLayout(self.f_c__layout)

        self.setContentsMargins(0, 0, 0, 0)

        self.spacer = QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.rvc_widget_view = QVBoxLayout(self)
        self.rvc_widget_view.setContentsMargins(0, 0, 0, 0)
        self.rvc_widget_view.addItem(self.spacer)
        self.rvc_widget_view.addWidget(self.gen_settings)
        self.rvc_widget_view.addWidget(self.r_and_sub)
        self.rvc_widget_view.addWidget(self.f_c_)
        self.settings_drawer.addWidget(RVCSettings())
        self.buttons_layout.addWidget(self.settings_button)
        self.help_drawer.addWidget(BulkRVCHelp(self))
        self.buttons_layout.addWidget(self.help_button)
        self.rvc_widget_view.addLayout(self.buttons_layout)


    def __onFolderCardClicked(self):
        """ download folder card clicked slot """
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Choose A Directory"), "./")
        if not folder or folder == "":
            return

        self.rvc_dir.value = folder
        self.rvc_dir_card.setContent(folder)


class BulkGenerationTableWidget(BaseBulkWidget):
    def __init__(self, parent: FallTalkApp):
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

        self.f_and_u = QGroupBox()
        self.f_and_u.setStyleSheet("border: none")
        self.f_and_u_layout = QHBoxLayout()
        self.f_and_u_layout.setContentsMargins(0, 0, 0, 0)
        self.f_and_u_layout.addWidget(self.upload_file_card, 1)
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
            self.tr('Use RVC Upscaler (Recommended for Untrained)'),
            cfg.rvc_enabled
        )

        self.upscaler_enabled = SwitchSettingCard(
            FIF.MEGAPHONE,
            self.tr('Super Resolution'),
            self.tr('Use Super Resolution Upscaler (Recommended)'),
            cfg.apbwe_enabled
        )

        self.upscaler_settings = QGroupBox()
        self.upscaler_settings.setStyleSheet("border: none")
        self.upscaler_settings_layout = QHBoxLayout()
        self.upscaler_settings_layout.setContentsMargins(0, 0, 0, 0)
        self.upscaler_settings_layout.addWidget(self.rvc_enabled, 2)
        self.upscaler_settings_layout.addWidget(self.upscaler_enabled, 2)
        self.upscaler_settings.setLayout(self.upscaler_settings_layout)

        self.gen_settings = QGroupBox()
        self.gen_settings.setStyleSheet("border: none")
        self.gen_settings_layout = QHBoxLayout(self.gen_settings)
        self.gen_settings_layout.setContentsMargins(0, 0, 0, 0)

        self.gen_settings_layout.addWidget(self.xwm_card, 2)
        self.gen_settings_layout.addWidget(self.delete_leftovers, 2)
        self.gen_settings.setLayout(self.gen_settings_layout)
        self.bulk_widget_view.addWidget(self.upscaler_settings)
        self.bulk_widget_view.addWidget(self.gen_settings)
        self.help_drawer.addWidget(BulkCSVHelp(self))
        self.buttons_layout.addWidget(self.help_button)
        self.bulk_widget_view.addLayout(self.buttons_layout)



class BulkGenerationWidget(FallTalkWidget):

    def __init__(self, parent: FallTalkApp):
        super().__init__(parent=parent, text="Bulk Generation", vertical=True)
        self.parent = parent

        self.pivot = SegmentedWidget(self)
        self.stackedWidget = QStackedWidget(self)

        # Create a TabView instance
        self.bulk_csv_widget = BulkGenerationTableWidget(parent)
        self.bulk_rvc_widget = BulkGenerationRVCWidget(parent)
        self.bulk_fuz_widget = BulkLipFuzWidget(parent)

        self.bulk_csv_widget.upload_file_card.clicked.connect(self.__onOutputFolderCardClicked)

        self.addSubInterface(self.bulk_csv_widget, 'bulk_csv_widget', 'CSV')
        self.addSubInterface(self.bulk_rvc_widget, 'bulk_rvc_widget', 'RVC')
        self.addSubInterface(self.bulk_fuz_widget, 'bulk_fuz_widget', 'FUZ')

        self.boxLayout.addWidget(self.pivot, 0, Qt.AlignmentFlag.AlignLeft)
        self.boxLayout.addWidget(self.stackedWidget)
        self.stackedWidget.currentChanged.connect(self.onCurrentIndexChanged)
        self.stackedWidget.setCurrentWidget(self.bulk_csv_widget)
        self.pivot.setCurrentItem(self.bulk_csv_widget.objectName())

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
