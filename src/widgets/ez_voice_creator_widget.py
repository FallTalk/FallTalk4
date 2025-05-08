from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

import csv
import os
import subprocess

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QGroupBox, QHeaderView, QAbstractItemView, QFileDialog
from qfluentwidgets import (
    FluentIcon as FIF, TableView, PushButton, PrimaryPushButton,
    ConfigItem, PushSettingCard, MessageBox, SwitchSettingCard, SearchLineEdit
)


from src.config.config import cfg, FileValidator
from src.utils.filesystem_utils import get_app_root
from src.widgets.falltalk_widget import FallTalkWidget
from src.widgets.table_models import TableModel


class EzVoiceCreatorWidget(FallTalkWidget):
    def __init__(self, parent: FallTalkApp):
        super().__init__(parent=parent, text="ESP Voice Generator", vertical=True)
        self.parent = parent

        # Table view
        self.dialogue_table = TableView()
        self.dialogue_table.setBorderVisible(True)
        self.dialogue_table.setBorderRadius(8)
        self.dialogue_table.setAlternatingRowColors(True)
        self.dialogue_table.setWordWrap(False)
        self.dialogue_table.verticalHeader().setVisible(False)
        self.dialogue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)  # Disable direct editing
        self.dialogue_table.setSortingEnabled(True)

        # Set up table headers
        self.headers = ["FILE_NAME", "RESPONSE TEXT", "VOICE TYPE", "FULLPATH", "REFERENCE FILE", "PLUGIN"]
        model = TableModel([], self.headers)
        self.dialogue_table.setModel(model)
        
        # Hide unnecessary columns
        self.dialogue_table.setColumnHidden(0, False)  # FILE_NAME
        self.dialogue_table.setColumnHidden(3, True)  # FULLPATH
        self.dialogue_table.setColumnHidden(4, False)  # REFERENCE FILE
        self.dialogue_table.setColumnHidden(5, False)  # PLUGIN
        
        # Set column resize modes
        self.dialogue_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # RESPONSE TEXT
        self.dialogue_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)  # VOICE TYPE
        
        # Update header text
        self.dialogue_table.model().setHeaderData(1, Qt.Orientation.Horizontal, "Text")

        # Filter search and edit mode controls
        self.controls_layout = QHBoxLayout()
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        
        self.filter_line_edit = SearchLineEdit()
        self.filter_line_edit.setPlaceholderText("Filter...")
        self.filter_line_edit.textChanged.connect(self.apply_filter)
        
        self.edit_mode_button = PushButton(text="Edit Mode")
        self.edit_mode_button.setIcon(FIF.EDIT)
        self.edit_mode_button.setCheckable(True)
        self.edit_mode_button.clicked.connect(self.toggle_edit_mode)
        
        self.controls_layout.addWidget(self.filter_line_edit, 1)
        self.controls_layout.addWidget(self.edit_mode_button)
        
        self.controls_widget = QWidget()
        self.controls_widget.setLayout(self.controls_layout)

        # File picker and guide
        self.csv_file = ConfigItem("ez_voice", "csv_file", "Please Select a CSV File", FileValidator())
        self.csv_file_card = PushSettingCard(
            self.tr('Select CSV File'),
            FIF.DOCUMENT,
            self.tr("Dialogue CSV File"),
            self.csv_file.value,
        )
        self.csv_file_card.clicked.connect(self.__onCSVFileClicked)

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
        self.f_and_u_layout.addWidget(self.csv_file_card, 1)
        self.f_and_u_layout.addWidget(self.started_card, 1)
        self.f_and_u.setLayout(self.f_and_u_layout)

        # Generation settings
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

        # Generation settings layout
        self.gen_settings = QGroupBox()
        self.gen_settings.setStyleSheet("border: none")
        self.gen_settings_layout = QHBoxLayout()
        self.gen_settings_layout.setContentsMargins(0, 0, 0, 0)
        self.gen_settings_layout.addWidget(self.xwm_card, 2)
        self.gen_settings_layout.addWidget(self.delete_leftovers, 2)
        self.gen_settings_layout.addWidget(self.rvc_enabled, 2)
        self.gen_settings.setLayout(self.gen_settings_layout)

        # Buttons layout
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        # Run xEdit button
        self.run_xedit_button = PushButton(text="xEdit")
        self.run_xedit_button.setIcon(FIF.COMMAND_PROMPT)
        self.run_xedit_button.clicked.connect(self.run_xedit)
        
        # Generate button
        self.generate_button = PrimaryPushButton(text="Generate Audio")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.generate_audio)
        
        self.buttons_layout.addWidget(self.run_xedit_button)
        self.buttons_layout.addWidget(self.generate_button)

        # Create a widget to hold the buttons layout
        self.buttons_widget = QWidget()
        self.buttons_widget.setLayout(self.buttons_layout)

        # Add widgets to frame
        self.addToFrame(self.dialogue_table)
        self.addToFrame(self.controls_widget)
        self.addToFrame(self.f_and_u)
        self.addToFrame(self.gen_settings)
        self.addToFrame(self.buttons_widget)

    def toggle_edit_mode(self, checked):
        if checked:
            self.dialogue_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
            self.edit_mode_button.setText("Editing...")
        else:
            self.dialogue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.edit_mode_button.setText("Edit Mode")

    def __onCSVFileClicked(self):
        allowed_file_types = "CSV files (*.csv)"
        file = QFileDialog.getOpenFileName(
            self, self.tr("Choose CSV File"), "./", allowed_file_types)
        if not file or file[0] == "":
            return

        # Clear the table before loading new data
        self.clear()
        
        self.csv_file.value = file[0]
        self.csv_file_card.setContent(file[0])
        
        # Load the CSV data immediately after selection
        self.load_csv()

    def load_csv(self):
        if not self.csv_file.value or self.csv_file.value == "Please Select a CSV File":
            MessageBox("Error", "Please select a CSV file first", self).exec()
            return

        try:
            data = []
            with open(self.csv_file.value, mode='r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    # Only include rows where VOICE TYPE matches a character's display_name (case insensitive)
                    voice_type = row['VOICE TYPE']
                    matching_character = None
                    for character in self.parent.characters_data.values():
                        if character['name'].lower() == voice_type.lower():
                            matching_character = character
                            break
                    
                    if matching_character:
                        # Use TOPIC TEXT if RESPONSE TEXT is null or empty
                        response_text = row['RESPONSE TEXT']
                        if not response_text or response_text.strip() == '':
                            response_text = row.get('TOPIC TEXT', '')

                        if response_text and response_text.strip() != '':
                            data.append([
                                row['FILENAME'],
                                response_text,
                                matching_character['name'],  # Use the actual display_name value
                                row['FULLPATH'],
                                "",  # Empty reference file by default
                                row.get('PLUGIN', '')  # Get PLUGIN or empty string if not present
                            ])

            model = TableModel(data, self.headers)
            self.dialogue_table.setModel(model)
            
            # Reapply column visibility and header text after model change
            self.dialogue_table.setColumnHidden(0, False)
            self.dialogue_table.setColumnHidden(3, True)
            self.dialogue_table.setColumnHidden(4, False)
            self.dialogue_table.setColumnHidden(5, False)
            self.dialogue_table.model().setHeaderData(1, Qt.Orientation.Horizontal, "Text")
            
        except Exception as e:
            MessageBox("Error", f"Failed to load CSV: {str(e)}", self).exec()

    def run_xedit(self):
        # Get xEdit path
        xedit_path = os.path.join(get_app_root(), "resource", "apps", "xedit", "xEdit64.exe")
        csv_path = os.path.join(get_app_root(), "Fallout4_DialogueExport.csv")

        script_path = os.path.join(get_app_root(), "resource", "apps", "xedit", "scripts", "Fallout4ExportDialogue.pas")
        if not os.path.exists(xedit_path):
            MessageBox("Error", "xEdit64.exe not found in resources folder", self).exec()
            return

        # Run xEdit
        try:
            subprocess.run([xedit_path, "-fo4", "-autoexit", "-autoload", f"-script:{script_path}"], check=True)
        except subprocess.CalledProcessError as e:
            MessageBox("Error", f"Failed to run xEdit: {str(e)}", self).exec()
        except Exception as e:
            MessageBox("Error", f"An error occurred: {str(e)}", self).exec()

        self.csv_file.value = csv_path
        self.csv_file_card.setContent(csv_path)

        # Load the CSV data immediately after selection
        self.load_csv()

    def generate_audio(self):
        """Handle generate button click"""
        self.parent.ez_voice_creator_inference()

    def clear(self):
        model = TableModel([], self.headers)
        self.dialogue_table.setModel(model)

    def apply_filter(self):
        text = self.filter_line_edit.text()
        model = self.dialogue_table.model()
        if model and text:
            for row in range(model.rowCount()):
                match = any(text.lower() in model.full_data(row, col).lower() for col in range(model.columnCount()))
                self.dialogue_table.setRowHidden(row, not match) 