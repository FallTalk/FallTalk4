from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp


import os
import re

import soundfile as sf
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtWidgets import QHBoxLayout, QStackedWidget, QHeaderView, QAbstractItemView, QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SearchLineEdit, TableView, PushButton, SegmentedWidget,
    CheckBox, ToolButton
)


from audio.audio_player import StandardAudioPlayerBar
from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.utils.audio_utils import extract_bsa, create_xwm, extract_fuz
from src.utils.filesystem_utils import get_app_root
from src.widgets.falltalk_widget import FallTalkWidget
from src.widgets.table_models import CustomReferencesModel, CustomTableModel
from src.widgets.drawer import RightDrawer
from src.help.reference_help import ReferencesHelp


class ReferencesWidget(FallTalkWidget):

    def __init__(self, parent: FallTalkApp):
        super().__init__(parent=parent, text="Reference Audio", vertical=True)
        self.parent = parent
        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVolume(100)

        self.pivot = SegmentedWidget(self)
        self.stackedWidget = QStackedWidget(self)

        self.reference_audio = []
        self.reference_audio_length = 0.0
        self.reference_transcripts = []

        # Initialize reference labels
        self.update_reference_labels()

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

        self.help_drawer = RightDrawer(self, title="About", icon=FIF.QUESTION)
        self.settings_drawer = RightDrawer(self, title="Advanced Settings", icon=FIF.SETTING)
        self.help_drawer.addWidget(ReferencesHelp(self))

        self.help_button = ToolButton()
        self.help_button.setIcon(FIF.QUESTION)
        self.help_button.setEnabled(True)
        self.help_button.clicked.connect(lambda: self.toggle_help_drawer())
        self.help_button.setFixedWidth(50)

        self.select_button = PushButton(text="Select")
        self.select_button.clicked.connect(self.select_row)
        self.select_button.setIcon(FIF.ADD_TO)
        self.remove_button = PushButton(text="Remove")
        self.remove_button.setIcon(FIF.REMOVE_FROM)
        self.remove_button.clicked.connect(self.remove_row)
        self.highlight_checkbox = CheckBox("Show Highlighted")
        self.highlight_checkbox.setMinimumWidth(120)
        self.highlight_checkbox.stateChanged.connect(self.apply_filter)
        self.controlsBox.addWidget(self.highlight_checkbox)
        self.controlsBox.addWidget(self.select_button, stretch=2)
        self.controlsBox.addWidget(self.remove_button, stretch=2)
        self.controlsBox.addWidget(self.help_button)
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
        # Clear all widgets that have a clear method
        for widget in self.parent.engine_widgets.values():
            if hasattr(widget, 'clear'):
                widget.clear()

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
            if os.path.exists(os.path.join(get_app_root(), f"temp/{filename}.wav")):
                QTimer.singleShot(0, lambda: (
                    player.setSource(QUrl.fromLocalFile(os.path.join(get_app_root(), f"temp/{filename}.wav")))
                ))
            else:
                QTimer.singleShot(0, lambda: (
                    extract_bsa(item),
                    extract_fuz(os.path.join(get_app_root(), f"temp/{filename}.fuz")),
                    create_xwm(os.path.join(get_app_root(), f"temp/{filename}.xwm"), os.path.join(get_app_root(), f"temp/{filename}.wav"), False),

                    os.path.exists(os.path.join(get_app_root(), f"temp/{filename}.xwm")) and os.remove(os.path.join(get_app_root(),f"temp/{filename}.xwm")),
                    os.path.exists(os.path.join(get_app_root(),f"temp/{filename}.fuz")) and os.remove(os.path.join(get_app_root(),f"temp/{filename}.fuz")),
                    os.path.exists(os.path.join(get_app_root(),f"temp/{filename}.lip")) and os.remove(os.path.join(get_app_root(),f"temp/{filename}.lip")),

                    player.setSource(QUrl.fromLocalFile(os.path.join(get_app_root(), f"temp/{filename}.wav")))
                ))

    def clear(self):
        self.reference_table.clearSelection()
        self.custom_reference_table.clearSelection()
        self.reference_audio = []
        self.reference_audio_length = 0
        self.reference_transcripts = []
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
        self.update_reference_labels()

    def select_row(self):
        index = self.stackedWidget.currentWidget().currentIndex()
        if index.isValid():
            # Call onReferenceSelect for all widgets that have the method
            for widget in self.parent.engine_widgets.values():
                if hasattr(widget, 'onReferenceSelect'):
                    widget.onReferenceSelect()

            row = index.row()
            model = self.stackedWidget.currentWidget().model()
            if (self.stackedWidget.currentWidget() == self.custom_reference_table):
                file_path = os.path.join(get_app_root(), f"references/{model.data(model.index(row, 0))}")
                if file_path not in self.reference_audio:
                    self.reference_audio.append(file_path)
                    self.increase(file_path)
                    model.toggle_selection(row)
                    # Custom references don't have transcripts in the table
                    self.reference_transcripts.append(None)
            else:
                filename = model.data(model.index(row, 0)).rsplit('.', 1)[0]
                file_path = os.path.join(get_app_root(), f"temp/{filename}.wav")
                if file_path not in self.reference_audio:
                    self.reference_audio.append(file_path)
                    self.increase(file_path)
                    model.toggle_selection(row)
                    # Get the transcript from the dialogue column
                    transcript = model.data(model.index(row, 1))
                    self.reference_transcripts.append(transcript)

    def increase(self, file_path):
        data, samplerate = sf.read(file_path)
        length_in_seconds = len(data) / samplerate
        self.reference_audio_length += length_in_seconds
        self.update_reference_labels()

    def remove_row(self):
        index = self.stackedWidget.currentWidget().currentIndex()
        if index.isValid():
            # Call onReferenceSelect for all widgets that have the method
            for widget in self.parent.engine_widgets.values():
                if hasattr(widget, 'onReferenceSelect'):
                    widget.onReferenceSelect()

            row = index.row()
            model = self.stackedWidget.currentWidget().model()
            if (self.stackedWidget.currentWidget() == self.custom_reference_table):
                file_path = os.path.join(get_app_root(), f"references/{model.data(model.index(row, 0))}")
                if file_path in self.reference_audio:
                    index = self.reference_audio.index(file_path)
                    self.reference_audio.remove(file_path)
                    if index < len(self.reference_transcripts):
                        self.reference_transcripts.pop(index)
                    model.toggle_selection(row)
                    self.decrease(file_path)
            else:
                filename = model.data(model.index(row, 0)).rsplit('.', 1)[0]
                file_path = os.path.join(get_app_root(), f"temp/{filename}.wav")
                if file_path in self.reference_audio:
                    index = self.reference_audio.index(file_path)
                    self.reference_audio.remove(file_path)
                    if index < len(self.reference_transcripts):
                        self.reference_transcripts.pop(index)
                    model.toggle_selection(row)
                    self.decrease(file_path)

    def decrease(self, file_path):
        data, samplerate = sf.read(file_path)
        length_in_seconds = len(data) / samplerate
        self.reference_audio_length -= length_in_seconds
        self.update_reference_labels()

    def _formatTime(self, time: float):
        s = int(time)
        ms = int((time - s) * 1000)
        ms_s = str(ms).zfill(2)[:2]
        return f'{s:02}:{ms_s}'

    def get_combined_transcript(self):
        """Combine all transcripts from selected references, leaving a space between each transcript."""
        # Filter out None values (from custom references)
        valid_transcripts = [t for t in self.reference_transcripts if t]
        if not valid_transcripts:
            return None
        # Join transcripts with a space between each
        return " ".join(valid_transcripts)

    def toggle_settings_drawer(self):
        self.settings_drawer.open_drawer()

    def toggle_help_drawer(self):
        self.help_drawer.open_drawer()

    def update_reference_labels(self):
        """Update reference label text and time label visibility based on whether references are selected."""
        if not self.reference_audio:
            # No references selected
            self.parent.reference_label.setText(self.tr("Reference:"))
            self.parent.reference_time_label.setVisible(True)
            self.parent.reference_time_label.setText(self.tr("Default"))
            # Reset style sheet when no references are selected
            self.parent.reference_label.setStyleSheet("")
        else:
            # References selected
            self.parent.reference_label.setText(self.tr("Reference:"))
            self.parent.reference_time_label.setVisible(True)
            self.parent.reference_time_label.setText(self.tr(self._formatTime(self.reference_audio_length)))

            # Default color if no engine or if engine doesn't have reference length requirements
            self.parent.reference_label.setStyleSheet("")

            # Color code the reference label based on the current engine's reference length requirements
            engine_type = EngineType(cfg.get(cfg.engine))
            if engine_type:
                min_length = engine_type.min_reference_length
                max_length = engine_type.max_reference_length

                # Set color based on reference length
                if min_length <= self.reference_audio_length <= max_length:
                    # Green: Within acceptable range
                    self.parent.reference_label.setStyleSheet("color: green;")
                elif self.reference_audio_length < min_length:
                    # Yellow: Too short
                    self.parent.reference_label.setStyleSheet("color: yellow;")
                elif self.reference_audio_length > max_length:
                    # Red: Too long
                    self.parent.reference_label.setStyleSheet("color: red;")
                return
