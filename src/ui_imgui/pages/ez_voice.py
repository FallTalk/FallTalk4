from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState

from src.config.config import cfg
from src.ui_imgui.page import Page
from src.ui_imgui.widgets.drawer import Drawer
from src.utils.filesystem_utils import get_app_root
from src.ui_imgui.widgets.common import draw_page_action_strip


class EzVoicePage(Page):
    page_id = "ez_voice"
    label = "ESP Voice Generator"
    icon = "fa-file-lines"
    nav_group = 2
    nav_position = "top"
    show_audio = False

    def __init__(self, state: AppState):
        self._state = state
        self._drawer = Drawer()
        self._ez_rows: list[list[str]] = []
        self._ez_headers: list[str] = ["FILE_NAME", "RESPONSE_TEXT", "VOICE_TYPE", "FULLPATH", "REFERENCE_FILE", "PLUGIN"]
        self._ez_filter = [""]
        self._edit_mode = [False]
        self._player_dialogue_only = [True]
        self._ck_mod_file = [""]
        self._ck_exe_file = [""]

    def draw(self):
        from imgui_bundle import imgui, icons_fontawesome_6 as fa
        from src.ui_imgui.widgets.tables import search_filter

        draw_page_action_strip(
            "ez_settings",
            "ez_help",
            "EZ Voice Settings",
            "EZ Voice Help",
            lambda: self._drawer.open(self._draw_settings_drawer, "EZ Voice Settings", fa.ICON_FA_GEAR),
            lambda: self._drawer.open(self._draw_help_drawer, "EZ Voice Help", fa.ICON_FA_CIRCLE_QUESTION),
        )

        if imgui.button("Import CSV##ez"):
            self._import_ez_csv()
        imgui.same_line()
        if imgui.button("xEdit##ez"):
            self._run_xedit()
        imgui.same_line()
        if imgui.button("Creation Kit##ez"):
            self._run_creation_kit()
        imgui.same_line()
        if imgui.button("Clear##ez"):
            self._ez_rows.clear()
        imgui.same_line()
        if imgui.button("Add Row##ez"):
            self._ez_rows.append([""] * len(self._ez_headers))
        imgui.spacing()
        search_filter("ez_search", self._ez_filter)
        imgui.same_line()
        changed, val = imgui.slider_int("Count##ez", cfg.get(cfg.ez_total), 1, 10)
        if changed:
            cfg.set(cfg.ez_total, val)
        imgui.same_line()
        changed, edit_mode = imgui.checkbox("Edit Mode##ez", self._edit_mode[0])
        if changed:
            self._edit_mode[0] = edit_mode
        imgui.same_line()
        changed, player_only = imgui.checkbox("Player Dialogue Only##ez", self._player_dialogue_only[0])
        if changed:
            self._player_dialogue_only[0] = player_only

        imgui.spacing()
        imgui.text_disabled("Post-Processing")
        changed, val = imgui.checkbox("RVC##ez", cfg.get(cfg.rvc_enabled))
        if changed:
            cfg.set(cfg.rvc_enabled, val)
        imgui.same_line()
        changed, val = imgui.checkbox("Super Resolution##ez", cfg.get(cfg.apbwe_enabled))
        if changed:
            cfg.set(cfg.apbwe_enabled, val)

        imgui.same_line()
        if imgui.button("Generate##ez"):
            self._start_ez_voice()

        imgui.separator()
        self._draw_editable_table("ez", self._ez_headers, self._ez_rows, self._ez_filter)

        self._drawer.draw()

    def _draw_settings_drawer(self):
        from imgui_bundle import imgui

        imgui.text("EZ Voice Settings")
        imgui.separator()
        changed, val = imgui.checkbox("RVC", cfg.get(cfg.rvc_enabled))
        if changed:
            cfg.set(cfg.rvc_enabled, val)
        changed, val = imgui.checkbox("Super Resolution", cfg.get(cfg.apbwe_enabled))
        if changed:
            cfg.set(cfg.apbwe_enabled, val)
        changed, val = imgui.checkbox("Edit Mode", self._edit_mode[0])
        if changed:
            self._edit_mode[0] = val
        changed, val = imgui.slider_int("Count", cfg.get(cfg.ez_total), 1, 10)
        if changed:
            cfg.set(cfg.ez_total, val)

    def _draw_help_drawer(self):
        from imgui_bundle import imgui

        imgui.text("EZ Voice Help")
        imgui.separator()
        imgui.text_wrapped("Import CSV or load dialogue export data from xEdit or Creation Kit.")
        imgui.text_wrapped("Use Edit Mode to make table cells editable, then generate the dataset with the selected count.")
        imgui.text_wrapped("xEdit exports Fallout4_DialogueExport.csv from the bundled script. Creation Kit exports dialogueExport.txt in the game directory.")

    def _import_ez_csv(self):
        from src.ui_imgui.widgets.file_dialog import open_file

        path = open_file("Import EzVoice CSV", [("CSV Files", "*.csv")])
        if path:
            self._load_csv_file(path)

    def _load_csv_file(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                if rows:
                    self._ez_headers = rows[0]
                    self._ez_rows = rows[1:]
        except Exception as e:
            self._state.error_queue.put(("CSV Error", str(e)))

    def _load_dialogue_file(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                if not reader.fieldnames:
                    raise ValueError("Dialogue export is missing headers.")
                required_columns = ["RESPONSE TEXT", "VOICE TYPE", "FILENAME", "FULL PATH"]
                missing_columns = [col for col in required_columns if col not in reader.fieldnames]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

                rows = []
                for row in reader:
                    response_text = row.get("RESPONSE TEXT", "")
                    if not response_text.strip():
                        response_text = row.get("TOPIC TEXT", "")
                    filename = row.get("FILENAME", "")
                    fullpath = row.get("FULL PATH", "").replace(".xwm", ".fuz")
                    voice_type = row.get("VOICE TYPE", "")
                    if response_text.strip() and filename:
                        rows.append([filename, response_text, voice_type, fullpath, "", row.get("PLUGIN", "")])
                self._ez_headers = ["FILE_NAME", "RESPONSE_TEXT", "VOICE_TYPE", "FULLPATH", "REFERENCE_FILE", "PLUGIN"]
                self._ez_rows = rows
        except Exception as e:
            self._state.error_queue.put(("CSV Error", str(e)))

    def _run_xedit(self):
        xedit_path = os.path.join(get_app_root(), "resource", "apps", "xedit", "xEdit64.exe")
        csv_path = os.path.join(get_app_root(), "Fallout4_DialogueExport.csv")
        script_path = os.path.join(get_app_root(), "resource", "apps", "xedit", "scripts", "Fallout4ExportDialogue.pas")

        if not os.path.exists(xedit_path):
            self._state.error_queue.put(("Error", "xEdit64.exe not found in resources folder."))
            return
        if os.path.exists(csv_path):
            os.remove(csv_path)

        try:
            subprocess.run([xedit_path, "-fo4", "-autoexit", "-autoload", f"-script:{script_path}"], check=True)
            if os.path.exists(csv_path):
                self._load_csv_file(csv_path)
            else:
                self._state.error_queue.put(("Error", "xEdit did not create Fallout4_DialogueExport.csv."))
        except subprocess.CalledProcessError as e:
            self._state.error_queue.put(("Error", f"Failed to run xEdit: {str(e)}"))
        except Exception as e:
            self._state.error_queue.put(("Error", f"An error occurred: {str(e)}"))

    def _run_creation_kit(self):
        from src.ui_imgui.widgets.file_dialog import open_file

        mod_file = open_file("Select Mod File", [("ESP Files", "*.esp"), ("ESL Files", "*.esl"), ("ESM Files", "*.esm")])
        if not mod_file:
            return
        ck_exe = open_file("Select Creation Kit Executable", [("Executable Files", "*.exe")])
        if not ck_exe:
            return
        self._ck_mod_file[0] = mod_file
        self._ck_exe_file[0] = ck_exe

        try:
            ck_dir = os.path.dirname(ck_exe)
            output_file = os.path.join(ck_dir, "dialogueExport.txt")
            local_output_file = os.path.join(get_app_root(), "dialogueExport.txt")
            if os.path.exists(local_output_file):
                os.remove(local_output_file)
            if os.path.exists(output_file):
                os.remove(output_file)

            bat_contents = f"""@echo off
cd /d "{ck_dir}"
"{ck_exe}" -ExportDialogue:{os.path.basename(mod_file)}
"""
            with tempfile.NamedTemporaryFile("w", suffix=".bat", delete=False, encoding="utf-8") as f:
                bat_file = f.name
                f.write(bat_contents)

            try:
                subprocess.run([bat_file], check=True)
            finally:
                if os.path.exists(bat_file):
                    os.remove(bat_file)

            timeout = 60
            check_interval = 1
            elapsed_time = 0
            while elapsed_time < timeout and not os.path.exists(output_file):
                time.sleep(check_interval)
                elapsed_time += check_interval

            if os.path.exists(output_file):
                shutil.copy2(output_file, local_output_file)
                self._load_dialogue_file(local_output_file)
            else:
                self._state.error_queue.put(("Error", "dialogueExport.txt was not created."))
        except subprocess.CalledProcessError as e:
            self._state.error_queue.put(("Error", f"Failed to run Creation Kit: {str(e)}"))
        except Exception as e:
            self._state.error_queue.put(("Error", f"An error occurred: {str(e)}"))

    def _start_ez_voice(self):
        if not self._ez_rows:
            self._state.error_queue.put(("Error", "No EzVoice data to generate."))
            return
        self._state.loading = True
        self._state.loading_message = "Starting EzVoice generation..."
        from src.utils.bulk_utils import ez_voice_generation

        threading.Thread(
            target=ez_voice_generation,
            args=(self._state.callbacks, self._ez_rows, self._ez_headers),
            daemon=True,
        ).start()

    def _draw_editable_table(self, table_id: str, headers: list[str], rows: list[list[str]], filter_text: list[str]):
        from imgui_bundle import imgui
        from src.ui_imgui.widgets.tables import begin_table, end_table

        all_cols = headers + ["Delete"]
        if begin_table(f"editable_{table_id}", all_cols):
            flt = filter_text[0].lower()
            voice_col = 2
            for idx, header in enumerate(headers):
                normalized = header.replace("_", " ").strip().upper()
                if normalized == "VOICE TYPE":
                    voice_col = idx
                    break
            to_delete = []
            for row_idx, row in enumerate(rows):
                if flt:
                    row_text = " ".join(row).lower()
                    if flt not in row_text:
                        continue
                if self._player_dialogue_only[0]:
                    voice_type = row[voice_col].lower() if voice_col < len(row) else ""
                    if voice_type not in {"playervoicemale01", "playervoicefemale01"}:
                        continue
                imgui.table_next_row()
                for col_idx, _header in enumerate(headers):
                    imgui.table_set_column_index(col_idx)
                    val = row[col_idx] if col_idx < len(row) else ""
                    if self._edit_mode[0]:
                        imgui.set_next_item_width(-1)
                        changed, new_val = imgui.input_text(f"##{table_id}_{row_idx}_{col_idx}", val)
                        if changed and col_idx < len(row):
                            row[col_idx] = new_val
                    else:
                        imgui.text_wrapped(val)
                imgui.table_set_column_index(len(headers))
                if self._edit_mode[0]:
                    if imgui.small_button(f"X##{table_id}_{row_idx}"):
                        to_delete.append(row_idx)
                else:
                    imgui.text_disabled("-")
            end_table()
            for idx in reversed(to_delete):
                rows.pop(idx)
