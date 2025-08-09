from __future__ import annotations

import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from src.ui.cards import SpinSettingCard
from src.utils.icons import FallTalkIcons

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

import csv
import os
import subprocess

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QGroupBox, QHeaderView, QAbstractItemView, QFileDialog, QVBoxLayout, QLabel, QDialog, QLineEdit
from qfluentwidgets import (
    FluentIcon as FIF, TableView, PushButton, PrimaryPushButton,
    ConfigItem, PushSettingCard, MessageBox, SwitchSettingCard, SearchLineEdit, ToolButton,
    CheckBox
)


from src.config.config import cfg, FileValidator
from src.utils.filesystem_utils import get_app_root
from src.widgets.falltalk_widget import FallTalkWidget
from src.widgets.table_models import TableModel
from src.help.ez_voice_creator_help import EzVoiceCreatorHelp
from src.widgets.drawer import RightDrawer

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
        self.filter_line_edit.setPlaceholderText("Only Generate Voices Matching Filter...")
        self.filter_line_edit.textChanged.connect(self.apply_filter)

        self.player_dialogue_checkbox = CheckBox("Player Dialogue Only")
        self.player_dialogue_checkbox.setChecked(True)
        self.player_dialogue_checkbox.setMinimumWidth(200)
        self.player_dialogue_checkbox.stateChanged.connect(self.apply_filter)

        self.edit_mode_button = PushButton(text="Edit Mode")
        self.edit_mode_button.setIcon(FIF.EDIT)
        self.edit_mode_button.setCheckable(True)
        self.edit_mode_button.clicked.connect(self.toggle_edit_mode)

        self.controls_layout.addWidget(self.filter_line_edit, 1)
        self.controls_layout.addWidget(self.player_dialogue_checkbox)
        self.controls_layout.addWidget(self.edit_mode_button)

        self.controls_widget = QWidget()
        self.controls_widget.setLayout(self.controls_layout)

        # File picker and guide
        self.csv_file = ConfigItem("ez_voice", "csv_file", "Please CSV File, Or Use xEdit", FileValidator())
        self.csv_file_card = PushSettingCard(
            self.tr('Select CSV File'),
            FIF.DOCUMENT,
            self.tr("xEdit CSV File"),
            self.csv_file.value,
        )
        self.csv_file_card.clicked.connect(self.__onCSVFileClicked)

        # Dialogue file card for tab-delimited format
        self.dialogue_file = ConfigItem("ez_voice", "dialogue_file", "Select File, Or Use Creation Kit", FileValidator())
        self.dialogue_file_card = PushSettingCard(
            self.tr('Select Dialogue File'),
            FIF.DOCUMENT,
            self.tr("Creation Kit Export File"),
            self.dialogue_file.value,
        )
        self.dialogue_file_card.clicked.connect(self.__onDialogueFileClicked)


        self.threads_card = SpinSettingCard(
            cfg.ez_total,
            FIF.STOP_WATCH,
            self.tr('Total Runs'),
            self.tr('How many times to generate the dataset?'),
            step=1
        )

        self.f_and_u = QGroupBox()
        self.f_and_u.setStyleSheet("border: none")
        self.f_and_u_layout = QHBoxLayout()
        self.f_and_u_layout.setContentsMargins(0, 0, 0, 0)
        self.f_and_u_layout.addWidget(self.dialogue_file_card, 1)
        self.f_and_u_layout.addWidget(self.csv_file_card, 1)
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
            self.tr('Use RVC Upscaler (Recommended for Untrained)'),
            cfg.rvc_enabled
        )

        # Generation settings layout
        self.gen_settings = QGroupBox()
        self.gen_settings.setStyleSheet("border: none")
        self.gen_settings_layout = QHBoxLayout()
        self.gen_settings_layout.setContentsMargins(0, 0, 0, 0)
        self.gen_settings_layout.addWidget(self.threads_card, 2)
        self.gen_settings_layout.addWidget(self.xwm_card, 2)
        self.gen_settings_layout.addWidget(self.delete_leftovers, 2)
        self.gen_settings.setLayout(self.gen_settings_layout)

        self.upscaler_enabled = SwitchSettingCard(
            FIF.MEGAPHONE,
            self.tr('Super Resolution'),
            self.tr('Use Super Resolution Upscaler (Recommended)'),
            cfg.apbwe_enabled
        )


        self.pad_short_phrases = SwitchSettingCard(
            FallTalkIcons.PADDING.icon(stroke=True),
            self.tr('Pad Short Phrases'),
            self.tr('Duplicate short phrases to improve quality, increases generation time'),
            cfg.pad_short_phrases
        )

        self.upscaler_settings = QGroupBox()
        self.upscaler_settings.setStyleSheet("border: none")
        self.upscaler_settings_layout = QHBoxLayout()
        self.upscaler_settings_layout.setContentsMargins(0, 0, 0, 0)
        self.upscaler_settings_layout.addWidget(self.rvc_enabled, 2)
        self.upscaler_settings_layout.addWidget(self.upscaler_enabled, 2)
        self.upscaler_settings_layout.addWidget(self.pad_short_phrases, 2)
        self.upscaler_settings.setLayout(self.upscaler_settings_layout)

        # Buttons layout
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)

        # Run xEdit button
        self.run_xedit_button = PushButton(text="xEdit")
        self.run_xedit_button.setIcon(FIF.COMMAND_PROMPT)
        self.run_xedit_button.clicked.connect(self.run_xedit)

        # Run CK button
        self.run_ck_button = PushButton(text="Creation Kit (recommended)")
        self.run_ck_button.setIcon(FallTalkIcons.BETHESDA.icon())
        self.run_ck_button.clicked.connect(self.run_ck)

        # Generate button
        self.generate_button = PrimaryPushButton(text="Generate Audio")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.generate_audio)
        self.generate_button.setEnabled(False)

        self.buttons_layout.addWidget(self.run_xedit_button)
        self.buttons_layout.addWidget(self.run_ck_button)
        self.buttons_layout.addWidget(self.generate_button)

        self.help_drawer = RightDrawer(self, title="About", icon=FIF.QUESTION)
        self.settings_drawer = RightDrawer(self, title="Advanced Settings", icon=FIF.SETTING)

        self.help_drawer.addWidget(EzVoiceCreatorHelp(self))
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

        self.buttons_layout.addWidget(self.help_button)

        # Create a widget to hold the buttons layout
        self.buttons_widget = QWidget()
        self.buttons_widget.setLayout(self.buttons_layout)



        # Add widgets to frame
        self.addToFrame(self.dialogue_table)
        self.addToFrame(self.controls_widget)
        self.addToFrame(self.f_and_u)
        self.addToFrame(self.gen_settings)
        self.addToFrame(self.upscaler_settings)
        self.addToFrame(self.buttons_widget)

    def toggle_edit_mode(self, checked):
        if checked:
            self.dialogue_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
            self.edit_mode_button.setText("Editing...")
        else:
            self.dialogue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.edit_mode_button.setText("Edit Mode")

    def __onCSVFileClicked(self):
        allowed_file_types = "CSV files (*.csv);;Text files (*.txt)"
        file = QFileDialog.getOpenFileName(
            self, self.tr("Choose File"), "./", allowed_file_types)
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
            self.generate_button.setEnabled(True)

            self.apply_filter()

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
        self.parent.ez_voice_creator_inference(cfg.get(cfg.ez_total))

    def clear(self):
        model = TableModel([], self.headers)
        self.dialogue_table.setModel(model)
        self.player_dialogue_checkbox.setChecked(True)
        self.generate_button.setEnabled(False)

    def apply_filter(self):
        text = self.filter_line_edit.text()
        player_dialogue_only = self.player_dialogue_checkbox.isChecked()
        model = self.dialogue_table.model()

        if model:
            for row in range(model.rowCount()):
                # Check text filter match
                text_match = True
                if text:
                    text_match = any(text.lower() in model.full_data(row, col).lower() for col in range(model.columnCount()))

                # Check player dialogue filter match
                player_match = True
                if player_dialogue_only:
                    voice_type = model.full_data(row, 2).lower()  # VOICE TYPE is at index 2
                    player_match = voice_type == "playervoicemale01" or voice_type == "playervoicefemale01"

                # Hide row if it doesn't match all active filters
                self.dialogue_table.setRowHidden(row, not (text_match and player_match))

    def toggle_settings_drawer(self):
        self.settings_drawer.open_drawer()

    def toggle_help_drawer(self):
        self.help_drawer.open_drawer()

    def __onDialogueFileClicked(self):
        allowed_file_types = "Text files (*.txt);;All files (*.*)"
        file = QFileDialog.getOpenFileName(
            self, self.tr("Choose Dialogue Export File"), "./", allowed_file_types)
        if not file or file[0] == "":
            return

        # Clear the table before loading new data
        self.clear()

        self.dialogue_file.value = file[0]
        self.dialogue_file_card.setContent(file[0])

        # Load the dialogue data immediately after selection
        self.load_dialogue()

    def run_ck(self):
        # Create a dialog to get Mod and CK location
        dialog = QDialog(self)
        dialog.setWindowTitle("Creation Kit Settings")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        # Mod file selection
        mod_label = QLabel("Mod File (ESP):")
        mod_path = QLineEdit()
        mod_path.setReadOnly(True)
        mod_browse = PushButton("Browse")

        mod_layout = QHBoxLayout()
        mod_layout.addWidget(mod_path, 1)
        mod_layout.addWidget(mod_browse)

        # CK executable selection
        ck_label = QLabel("Creation Kit Location:")
        ck_path = QLineEdit()
        ck_path.setReadOnly(True)
        ck_browse = PushButton("Browse")

        ck_layout = QHBoxLayout()
        ck_layout.addWidget(ck_path, 1)
        ck_layout.addWidget(ck_browse)

        # Run button
        run_button = PrimaryPushButton("Run")
        run_button.setEnabled(False)

        # Add widgets to layout
        layout.addWidget(mod_label)
        layout.addLayout(mod_layout)
        layout.addWidget(ck_label)
        layout.addLayout(ck_layout)
        layout.addWidget(run_button)

        # Connect signals
        def browse_mod():
            file = QFileDialog.getOpenFileName(dialog, "Select Mod File", cfg.get(cfg.fallout_4_directory), "ESP Files (*.esp)")
            if file and file[0]:
                mod_path.setText(file[0])
                update_run_button()

        def browse_ck():
            file = QFileDialog.getOpenFileName(dialog, "Select Creation Kit Executable",  cfg.get(cfg.fallout_4_directory), "Executable Files (*.exe)")
            if file and file[0]:
                ck_path.setText(file[0])
                update_run_button()

        def update_run_button():
            run_button.setEnabled(mod_path.text() is not None and ck_path.text() is not None)

        def run_ck_export():
            mod_file = mod_path.text()
            ck_exe = ck_path.text()

            if not mod_file or not ck_exe:
                MessageBox("Error", "Please select both Mod file and Creation Kit executable", dialog).exec()
                return

            dialog.accept()

            try:
                ck_dir = os.path.dirname(ck_exe)
                output_file = os.path.join(ck_dir, "dialogueExport.txt")
                local_output_file = os.path.join(get_app_root(), "dialogueExport.txt")

                bat_contents = f"""@echo off
                cd /d "{ck_dir}"
                "{ck_exe}" -ExportDialogue:{os.path.basename(mod_file)}
                """
                with tempfile.NamedTemporaryFile("w", suffix=".bat", delete=False, encoding="utf-8") as f:
                    bat_file = f.name
                    f.write(bat_contents)

                try:
                    # Run .bat blocking
                    subprocess.run([bat_file], check=True)
                finally:
                    time.sleep(2)
                    # Clean up the temp .bat file
                    if os.path.exists(bat_file):
                        os.remove(bat_file)
                    shutil.copy2(output_file, local_output_file)

                # Check if the output file was created
                if os.path.exists(local_output_file):
                    self.dialogue_file.value = local_output_file
                    self.dialogue_file_card.setContent(local_output_file)

                    # Load the dialogue data
                    self.load_dialogue()
                else:
                    traceback.print_exc()
                    MessageBox("Error", "dialogueExport.txt was not created", self).exec()
            except subprocess.CalledProcessError as e:
                traceback.print_exc()
                MessageBox("Error", f"Failed to run Creation Kit: {str(e)}", self).exec()
            except Exception as e:
                traceback.print_exc()

                MessageBox("Error", f"An error occurred: {str(e)}", self).exec()

        mod_browse.clicked.connect(browse_mod)
        ck_browse.clicked.connect(browse_ck)
        run_button.clicked.connect(run_ck_export)

        dialog.exec()

    def load_dialogue(self):
        if not self.dialogue_file.value or self.dialogue_file.value == "Please Select a Dialogue Export File, Or Use CK below":
            MessageBox("Error", "Please select a dialogue export file first", self).exec()
            return

        try:
            data = []
            with open(self.dialogue_file.value, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter='\t')

                # Check if required columns exist
                required_columns = ['RESPONSE TEXT', 'VOICE TYPE', 'FILENAME', 'FULL PATH']
                missing_columns = [col for col in required_columns if col not in reader.fieldnames]

                if missing_columns:
                    MessageBox("Error", f"Missing required columns: {', '.join(missing_columns)}", self).exec()
                    return

                # Process the rest of the file
                for row in reader:
                    voice_type = row['VOICE TYPE']
                    response_text = row['RESPONSE TEXT']
                    filename = row['FILENAME']
                    fullpath = row['FULL PATH']

                    fullpath = fullpath.replace(".xwm", ".fuz")

                    p = Path(fullpath)
                    parts = p.parts
                    voice_index = parts.index('Voice')
                    plugin = parts[voice_index + 1]

                    # Only include rows where VOICE TYPE matches a character's display_name (case insensitive)
                    matching_character = None
                    for character in self.parent.characters_data.values():
                        if character['name'].lower() == voice_type.lower():
                            matching_character = character
                            break

                    if matching_character and response_text and response_text.strip() != '':
                        data.append([
                            filename,
                            response_text,
                            matching_character['name'],  # Use the actual display_name value
                            fullpath,
                            "",  # Empty reference file by default
                            plugin
                        ])

            model = TableModel(data, self.headers)
            self.dialogue_table.setModel(model)

            # Reapply column visibility and header text after model change
            self.dialogue_table.setColumnHidden(0, False)
            self.dialogue_table.setColumnHidden(3, True)
            self.dialogue_table.setColumnHidden(4, False)
            self.dialogue_table.setColumnHidden(5, False)
            self.dialogue_table.model().setHeaderData(1, Qt.Orientation.Horizontal, "Text")
            self.generate_button.setEnabled(True)

            self.apply_filter()

        except Exception as e:
            traceback.print_exc()
            MessageBox("Error", f"Failed to load dialogue file: {str(e)}", self).exec()
