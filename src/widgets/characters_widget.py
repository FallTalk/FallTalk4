import json
import os
import re
import shutil

from PySide6.QtCore import QTimer
from PySide6.QtGui import Qt
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QVBoxLayout, QStackedWidget, QHeaderView, QWidget
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import (
    SearchLineEdit, TableView, PushButton, SegmentedWidget,
    IconWidget, MessageBox, CheckBox
)

from src.config.config import cfg, CUSTOM_DISCLAIMER
from src.utils.icons import FallTalkStrokeIcons
from src.widgets.custom_message_box import CustomMessageBox
from src.widgets.table_models import CharacterTableModel
from src.widgets import FallTalkWidget


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

        model = self.custom_table.model()
        if model:
            for row in range(model.rowCount()):
                match = text.lower() in model.full_data(row, 0).lower() or text.lower() in model.full_data(row, 1).lower()
                if state:
                    rvc_match = model.full_data(row, 5)['RVC'] is not None
                    self.custom_table.setRowHidden(row, not match or not rvc_match)
                else:
                    self.custom_table.setRowHidden(row, not match)

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
