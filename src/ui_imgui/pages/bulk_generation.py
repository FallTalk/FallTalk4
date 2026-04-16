from __future__ import annotations

import csv
import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState

from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.ui_imgui.page import Page
from src.ui_imgui.widgets.common import (
    draw_page_action_strip,
    pop_accent_button_style,
    push_accent_button_style,
)
from src.ui_imgui.widgets.drawer import Drawer


class BulkGenerationPage(Page):
    page_id = "bulk_generation"
    label = "Bulk Generation"
    icon = "fa-bolt"
    nav_group = 2
    nav_position = "top"
    show_audio = False

    _CSV_HEADERS = ["FILE_NAME", "CHARACTER", "TEXT", "REFERENCE", "OUTPUT_DIR"]
    _CSV_LABELS = {
        "FILE_NAME": "File Name",
        "CHARACTER": "Character",
        "TEXT": "Text",
        "REFERENCE": "Reference",
        "OUTPUT_DIR": "Output Directory",
    }

    def __init__(self, state: AppState):
        self._state = state
        self._drawer = Drawer()
        self._csv_rows: list[list[str]] = []
        self._csv_headers: list[str] = list(self._CSV_HEADERS)
        self._csv_filter = [""]
        self._csv_selected_row = -1
        self._csv_source_path = ""
        self._rvc_folder = [""]
        self._rvc_char_idx = [0]
        self._fuz_folder = [""]

    def _get_rvc_characters(self) -> list[dict]:
        characters = []
        for source in (self._state.models or {}, self._state.custom_models or {}):
            for name, model_data in source.items():
                if isinstance(model_data, dict) and model_data.get("RVC"):
                    entry = dict(model_data)
                    entry.setdefault("name", name)
                    entry.setdefault("display_name", model_data.get("display_name", name))
                    characters.append(entry)
        return sorted(characters, key=lambda item: item.get("display_name", item.get("name", "")).lower())

    def draw(self):
        from imgui_bundle import icons_fontawesome_6 as fa, imgui

        draw_page_action_strip(
            "bulk_settings",
            "bulk_help",
            "Bulk Settings",
            "Bulk Help",
            lambda: self._drawer.open(self._draw_settings_drawer, "Bulk Settings", fa.ICON_FA_GEAR),
            lambda: self._drawer.open(self._draw_help_drawer, "Bulk Help", fa.ICON_FA_CIRCLE_QUESTION),
        )

        if imgui.begin_tab_bar("bulk_tabs"):
            if imgui.begin_tab_item("CSV Generation")[0]:
                self._draw_csv_panel()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Bulk RVC")[0]:
                self._draw_rvc_panel()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Bulk FUZ")[0]:
                self._draw_fuz_panel()
                imgui.end_tab_item()
            imgui.end_tab_bar()

        self._drawer.draw()

    def _draw_settings_drawer(self):
        from imgui_bundle import imgui

        imgui.text("Bulk Generation Settings")
        imgui.separator()
        changed, val = imgui.checkbox("RVC##bulk_settings", cfg.get(cfg.rvc_enabled))
        if changed:
            cfg.set(cfg.rvc_enabled, val)
        changed, val = imgui.checkbox("Super Resolution##bulk_settings", cfg.get(cfg.apbwe_enabled))
        if changed:
            cfg.set(cfg.apbwe_enabled, val)
        changed, val = imgui.checkbox("Include Subdirectories##bulk_settings", cfg.get(cfg.include_subdir))
        if changed:
            cfg.set(cfg.include_subdir, val)
        changed, val = imgui.checkbox("Replace Existing##bulk_settings", cfg.get(cfg.replace_existing))
        if changed:
            cfg.set(cfg.replace_existing, val)
        changed, val = imgui.checkbox("Use Existing LIP##bulk_settings", cfg.get(cfg.use_existing_lip))
        if changed:
            cfg.set(cfg.use_existing_lip, val)
        changed, val = imgui.slider_int("Threads##bulk_settings", cfg.get(cfg.threads), 1, 16)
        if changed:
            cfg.set(cfg.threads, val)

    def _draw_help_drawer(self):
        from imgui_bundle import imgui

        imgui.text("Bulk Generation")
        imgui.separator()
        imgui.text_wrapped(
            "CSV Generation expects rows for file name, character, text, reference, and output directory. "
            "You can import a header-based CSV or a raw row file and then correct any flagged rows in the editor."
        )
        imgui.text_wrapped("Bulk RVC converts audio files in a folder using the selected RVC-capable character model.")
        imgui.text_wrapped("Bulk FUZ builds LIP/FUZ outputs from a folder and can reuse existing LIP files.")

    def _ensure_csv_selection(self):
        if not self._csv_rows:
            self._csv_selected_row = -1
            return
        if self._csv_selected_row < 0 or self._csv_selected_row >= len(self._csv_rows):
            self._csv_selected_row = 0

    @staticmethod
    def _blank_csv_row() -> list[str]:
        return ["", "", "", "", ""]

    @staticmethod
    def _sanitize_row(row: list[str]) -> list[str]:
        clean = [(value or "").strip() for value in row[:5]]
        while len(clean) < 5:
            clean.append("")
        return clean

    def _row_is_blank(self, row: list[str]) -> bool:
        return not any((value or "").strip() for value in row)

    def _row_issues(self, row: list[str]) -> list[str]:
        clean = self._sanitize_row(row)
        if self._row_is_blank(clean):
            return ["Blank row"]

        issues = []
        if not clean[1]:
            issues.append("Missing character")
        if not clean[2]:
            issues.append("Missing text")
        return issues

    def _visible_csv_rows(self):
        query = self._csv_filter[0].strip().lower()
        for idx, row in enumerate(self._csv_rows):
            clean = self._sanitize_row(row)
            if query and query not in " ".join(clean).lower():
                continue
            yield idx, clean

    @classmethod
    def _display_label(cls, header: str) -> str:
        return cls._CSV_LABELS.get(header, header.replace("_", " ").title())

    def _draw_csv_panel(self):
        from imgui_bundle import imgui
        from src.ui_imgui.widgets.tables import search_filter

        if self._state.engine_type == EngineType.RVC:
            imgui.text_disabled("CSV Generation")
            imgui.separator()
            imgui.text_wrapped(
                "Bulk CSV generation is disabled while the active engine is RVC. "
                "RVC does not support the normal text-to-speech batch workflow."
            )
            return

        self._ensure_csv_selection()
        visible_rows = list(self._visible_csv_rows())
        actionable_rows = [row for row in self._csv_rows if not self._row_is_blank(row)]
        invalid_rows = [
            idx
            for idx, row in enumerate(self._csv_rows)
            if not self._row_is_blank(row) and self._row_issues(row)
        ]
        ready_rows = max(0, len(actionable_rows) - len(invalid_rows))

        imgui.begin_child(
            "##bulk_csv_overview",
            size=imgui.ImVec2(0, 116),
            child_flags=imgui.ChildFlags_.borders,
        )
        try:
            imgui.text_disabled("Guided CSV Workflow")
            imgui.separator()
            imgui.text_wrapped(
                "Expected columns: File Name, Character, Text, Reference, Output Directory. "
                "Imports can include headers or raw rows; blank file names and output directories are filled automatically."
            )
            imgui.text(f"Rows: {len(actionable_rows)}")
            imgui.same_line()
            imgui.text(f"Ready: {ready_rows}")
            imgui.same_line()
            if invalid_rows:
                imgui.text_colored((0.96, 0.75, 0.33, 1.0), f"Needs Attention: {len(invalid_rows)}")
            else:
                imgui.text_colored((0.48, 0.84, 0.54, 1.0), "All visible rows are ready.")
            if self._csv_source_path:
                imgui.text_disabled(f"Imported From: {self._csv_source_path}")
        finally:
            imgui.end_child()

        imgui.spacing()
        if imgui.button("Import CSV / TXT##csv_import"):
            self._import_csv()
        imgui.same_line()
        if imgui.button("Add Row##csv_add"):
            self._csv_rows.append(self._blank_csv_row())
            self._csv_selected_row = len(self._csv_rows) - 1
        imgui.same_line()
        if imgui.button("Duplicate Row##csv_duplicate"):
            self._duplicate_selected_row()
        imgui.same_line()
        if imgui.button("Delete Row##csv_delete"):
            self._delete_selected_row()
        imgui.same_line()
        if imgui.button("Clear Table##csv_clear"):
            self._csv_rows.clear()
            self._csv_selected_row = -1
        imgui.same_line()
        search_filter("csv_search", self._csv_filter, width=200.0)
        imgui.same_line()
        push_accent_button_style()
        try:
            if imgui.button("Generate Batch##csv_generate"):
                self._start_bulk_csv()
        finally:
            pop_accent_button_style()

        imgui.spacing()
        imgui.text_disabled("Post-Processing")
        changed, val = imgui.checkbox("RVC##csv", cfg.get(cfg.rvc_enabled))
        if changed:
            cfg.set(cfg.rvc_enabled, val)
        imgui.same_line()
        changed, val = imgui.checkbox("Super Resolution##csv", cfg.get(cfg.apbwe_enabled))
        if changed:
            cfg.set(cfg.apbwe_enabled, val)

        imgui.separator()
        self._draw_csv_table(visible_rows)
        imgui.spacing()
        self._draw_csv_editor()

    def _draw_csv_table(self, visible_rows: list[tuple[int, list[str]]]):
        from imgui_bundle import imgui
        from src.ui_imgui.widgets.tables import begin_table, end_table

        cols = ["Edit", "File Name", "Character", "Text", "Reference", "Output Directory", "Status"]
        table_flags = (
            imgui.TableFlags_.resizable
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.scroll_y
        )
        if begin_table("bulk_csv_table", cols, flags=table_flags):
            selected_bg = imgui.get_color_u32(imgui.ImVec4(0.32, 0.28, 0.16, 1.0))
            for row_idx, row in visible_rows:
                issues = self._row_issues(row)
                is_selected = row_idx == self._csv_selected_row

                imgui.table_next_row()
                if is_selected:
                    imgui.table_set_bg_color(imgui.TableBgTarget_.row_bg0, selected_bg)

                imgui.table_set_column_index(0)
                if imgui.small_button(f"Edit##csv_edit_{row_idx}"):
                    self._csv_selected_row = row_idx

                for col_idx, value in enumerate(row):
                    imgui.table_set_column_index(col_idx + 1)
                    display = value if len(value) <= 64 else f"{value[:61]}..."
                    if col_idx == 2:
                        imgui.text_wrapped(display)
                    else:
                        imgui.text(display or "--")
                    if value and imgui.is_item_hovered():
                        imgui.set_tooltip(value)

                imgui.table_set_column_index(6)
                if issues:
                    imgui.text_colored((0.96, 0.75, 0.33, 1.0), ", ".join(issues))
                else:
                    imgui.text_colored((0.48, 0.84, 0.54, 1.0), "Ready")
            end_table()

        if not visible_rows:
            from imgui_bundle import imgui

            imgui.text_disabled("No rows match the current filter.")

    def _draw_csv_editor(self):
        from imgui_bundle import imgui
        from src.ui_imgui.widgets.file_dialog import open_folder

        self._ensure_csv_selection()

        imgui.begin_child(
            "##bulk_csv_editor",
            size=imgui.ImVec2(0, 244),
            child_flags=imgui.ChildFlags_.borders,
        )
        try:
            imgui.text_disabled("Row Editor")
            imgui.separator()

            if self._csv_selected_row < 0 or self._csv_selected_row >= len(self._csv_rows):
                imgui.text_wrapped("Add or import rows, then choose Edit on a row to refine its values here.")
                return

            row = self._sanitize_row(self._csv_rows[self._csv_selected_row])
            self._csv_rows[self._csv_selected_row] = row

            changed, row[0] = imgui.input_text("File Name##csv_editor_file", row[0])
            changed, row[1] = imgui.input_text("Character##csv_editor_character", row[1])
            imgui.text("Text")
            _, row[2] = imgui.input_text_multiline("##csv_editor_text", row[2], size=(0, 92))
            changed, row[3] = imgui.input_text("Reference##csv_editor_reference", row[3])
            changed, row[4] = imgui.input_text("Output Directory##csv_editor_output", row[4])
            imgui.same_line()
            if imgui.button("Browse##csv_editor_output"):
                folder = open_folder("Select Output Folder")
                if folder:
                    row[4] = folder

            issues = self._row_issues(row)
            if issues:
                imgui.text_colored((0.96, 0.75, 0.33, 1.0), "Needs Attention")
                imgui.text_wrapped(", ".join(issues))
            else:
                imgui.text_colored((0.48, 0.84, 0.54, 1.0), "Ready for generation")
                imgui.text_wrapped(
                    "File name and output directory can stay blank if you want FallTalk to create them automatically."
                )
        finally:
            imgui.end_child()

    def _duplicate_selected_row(self):
        self._ensure_csv_selection()
        if self._csv_selected_row < 0:
            return
        duplicate = list(self._sanitize_row(self._csv_rows[self._csv_selected_row]))
        self._csv_rows.insert(self._csv_selected_row + 1, duplicate)
        self._csv_selected_row += 1

    def _delete_selected_row(self):
        self._ensure_csv_selection()
        if self._csv_selected_row < 0:
            return
        self._csv_rows.pop(self._csv_selected_row)
        if not self._csv_rows:
            self._csv_selected_row = -1
        else:
            self._csv_selected_row = min(self._csv_selected_row, len(self._csv_rows) - 1)

    def _import_csv(self):
        from src.ui_imgui.widgets.file_dialog import open_file

        path = open_file(
            "Import CSV or Text",
            [("CSV Files", "*.csv"), ("Text Files", "*.txt"), ("All Files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as handle:
                parsed_rows = list(csv.reader(handle))
            headers, rows = self._parse_import_rows(path, parsed_rows)
            self._csv_headers = headers
            self._csv_rows = rows
            self._csv_source_path = path
            self._csv_selected_row = 0 if self._csv_rows else -1
        except Exception as exc:
            self._state.error_queue.put(("CSV Error", str(exc)))

    def _parse_import_rows(self, path: str, parsed_rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
        if not parsed_rows:
            raise ValueError("The selected file is empty.")

        data_rows = [self._sanitize_row(row) for row in parsed_rows if any((cell or "").strip() for cell in row)]
        if not data_rows:
            raise ValueError("The selected file does not contain any usable rows.")

        if path.lower().endswith(".txt") or not self._looks_like_header(data_rows[0]):
            return list(self._CSV_HEADERS), data_rows

        return self._normalize_bulk_csv(data_rows[0], data_rows[1:])

    @classmethod
    def _looks_like_header(cls, row: list[str]) -> bool:
        header_map = cls._header_map()
        matches = 0
        for value in row:
            if (value or "").strip().upper() in header_map:
                matches += 1
        return matches >= 2

    @classmethod
    def _header_map(cls) -> dict[str, str]:
        return {
            "FILE_NAME": "FILE_NAME",
            "FILENAME": "FILE_NAME",
            "FILE": "FILE_NAME",
            "CHARACTER": "CHARACTER",
            "VOICE_TYPE": "CHARACTER",
            "TEXT": "TEXT",
            "RESPONSE_TEXT": "TEXT",
            "REFERENCE": "REFERENCE",
            "REFERENCE_FILE": "REFERENCE",
            "OUTPUT_DIR": "OUTPUT_DIR",
            "FULLPATH": "OUTPUT_DIR",
        }

    @classmethod
    def _normalize_bulk_csv(cls, headers: list[str], rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
        source_lookup = {}
        for idx, header in enumerate(headers):
            key = cls._header_map().get((header or "").strip().upper())
            if key:
                source_lookup[key] = idx

        if "CHARACTER" not in source_lookup or "TEXT" not in source_lookup:
            raise ValueError("CSV headers must include character and text columns.")

        normalized_rows = []
        for row in rows:
            if not any((cell or "").strip() for cell in row):
                continue
            normalized_row = []
            for key in cls._CSV_HEADERS:
                source_idx = source_lookup.get(key)
                normalized_row.append(row[source_idx].strip() if source_idx is not None and source_idx < len(row) else "")
            normalized_rows.append(normalized_row)
        return list(cls._CSV_HEADERS), normalized_rows

    def _start_bulk_csv(self):
        from src.utils.bulk_utils import bulk_inference

        if self._state.engine_type == EngineType.RVC:
            self._state.error_queue.put(
                ("Error", "Bulk CSV generation is unavailable while the active engine is RVC.")
            )
            return

        if self._state.tts_engine is None:
            self._state.error_queue.put(("Error", "Load a character model before starting bulk generation."))
            return

        rows = [self._sanitize_row(row) for row in self._csv_rows if not self._row_is_blank(row)]
        if not rows:
            self._state.error_queue.put(("Error", "No CSV data to generate."))
            return

        invalid = [
            idx
            for idx, row in enumerate(self._csv_rows)
            if not self._row_is_blank(row) and self._row_issues(row)
        ]
        if invalid:
            self._csv_selected_row = invalid[0]
            self._state.error_queue.put(
                ("CSV Error", f"Fix the highlighted CSV rows before generating. First issue is row {invalid[0] + 1}.")
            )
            return

        self._state.loading = True
        self._state.loading_message = "Starting bulk CSV generation..."
        threading.Thread(
            target=bulk_inference,
            args=(self._state.callbacks, rows, self._csv_headers),
            daemon=True,
        ).start()

    def _draw_rvc_panel(self):
        from imgui_bundle import imgui

        imgui.text_disabled("Bulk RVC")
        imgui.separator()
        imgui.text("Input Folder")
        imgui.same_line()
        imgui.set_next_item_width(320)
        _, self._rvc_folder[0] = imgui.input_text("##rvc_folder", self._rvc_folder[0])
        imgui.same_line()
        if imgui.button("Browse##rvc"):
            from src.ui_imgui.widgets.file_dialog import open_folder

            folder = open_folder("Select Audio Folder")
            if folder:
                self._rvc_folder[0] = folder

        characters = self._get_rvc_characters()
        if characters:
            labels = [entry.get("display_name", entry.get("name", "Unknown")) for entry in characters]
            self._rvc_char_idx[0] = min(self._rvc_char_idx[0], len(labels) - 1)
            imgui.text("Character")
            imgui.same_line()
            imgui.set_next_item_width(220)
            _, self._rvc_char_idx[0] = imgui.combo("##rvc_char", self._rvc_char_idx[0], labels)
        else:
            imgui.text_disabled("No trained RVC models were found.")

        changed, val = imgui.checkbox("Include Subdirectories##rvc", cfg.get(cfg.include_subdir))
        if changed:
            cfg.set(cfg.include_subdir, val)
        changed, val = imgui.checkbox("Replace Existing##rvc", cfg.get(cfg.replace_existing))
        if changed:
            cfg.set(cfg.replace_existing, val)
        changed, val = imgui.checkbox("Use Existing LIP##rvc", cfg.get(cfg.use_existing_lip))
        if changed:
            cfg.set(cfg.use_existing_lip, val)
        changed, val = imgui.slider_int("Threads##rvc", cfg.get(cfg.threads), 1, 16)
        if changed:
            cfg.set(cfg.threads, val)

        push_accent_button_style()
        try:
            if imgui.button("Generate RVC Batch##rvc_generate"):
                self._start_bulk_rvc()
        finally:
            pop_accent_button_style()

    def _start_bulk_rvc(self):
        folder = self._rvc_folder[0]
        if not folder or not os.path.isdir(folder):
            self._state.error_queue.put(("Error", "Please select a valid folder."))
            return
        characters = self._get_rvc_characters()
        if not characters:
            self._state.error_queue.put(("Error", "No RVC-capable character models are available."))
            return
        self._state.loading = True
        self._state.loading_message = "Starting bulk RVC..."
        model = characters[self._rvc_char_idx[0]]
        from src.utils.bulk_utils import bulk_rvc_inference

        threading.Thread(
            target=bulk_rvc_inference,
            args=(self._state.callbacks, folder, model),
            daemon=True,
        ).start()

    def _draw_fuz_panel(self):
        from imgui_bundle import imgui

        imgui.text_disabled("Bulk FUZ")
        imgui.separator()
        imgui.text("Input Folder")
        imgui.same_line()
        imgui.set_next_item_width(320)
        _, self._fuz_folder[0] = imgui.input_text("##fuz_folder", self._fuz_folder[0])
        imgui.same_line()
        if imgui.button("Browse##fuz"):
            from src.ui_imgui.widgets.file_dialog import open_folder

            folder = open_folder("Select Folder")
            if folder:
                self._fuz_folder[0] = folder

        changed, val = imgui.checkbox("Include Subdirectories##fuz", cfg.get(cfg.include_subdir))
        if changed:
            cfg.set(cfg.include_subdir, val)
        changed, val = imgui.checkbox("Use Existing LIP##fuz", cfg.get(cfg.use_existing_lip))
        if changed:
            cfg.set(cfg.use_existing_lip, val)
        changed, val = imgui.slider_int("Threads##fuz", cfg.get(cfg.threads), 1, 16)
        if changed:
            cfg.set(cfg.threads, val)

        push_accent_button_style()
        try:
            if imgui.button("Generate FUZ Batch##fuz_generate"):
                self._start_bulk_fuz()
        finally:
            pop_accent_button_style()

    def _start_bulk_fuz(self):
        folder = self._fuz_folder[0]
        if not folder or not os.path.isdir(folder):
            self._state.error_queue.put(("Error", "Please select a valid folder."))
            return
        self._state.loading = True
        self._state.loading_message = "Starting bulk FUZ..."
        from src.utils.bulk_utils import bulk_fuz

        threading.Thread(
            target=bulk_fuz,
            args=(self._state.callbacks, folder),
            daemon=True,
        ).start()
