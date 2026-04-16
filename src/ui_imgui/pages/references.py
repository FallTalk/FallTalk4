from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState

from src.enums.engine_type import EngineType
from src.ui_imgui.page import Page
from src.ui_imgui.widgets.common import (
    get_accent_color,
    pop_accent_button_style,
    push_accent_button_style,
)
from src.utils.filesystem_utils import get_app_root

logger = logging.getLogger("falltalk.references")


def _resolve_reference_path(ref) -> str:
    """Resolve a reference audio entry to a playable .wav file path.

    For plain strings (custom references), returns the path directly.
    For Fallout 4 voice dicts, extracts from BSA archive if needed.
    """
    if isinstance(ref, str):
        return ref

    if not isinstance(ref, dict):
        return ""

    if ref.get("path"):
        return ref["path"]

    filename = ref.get("filename", "")
    if not filename:
        return ""

    stem = filename.rsplit(".", 1)[0]
    wav_path = os.path.join(get_app_root(), "temp", f"{stem}.wav")
    if os.path.exists(wav_path):
        return wav_path

    from src.config.config import cfg

    if cfg.get(cfg.fallout_4_directory) == "fallout4.exe not found":
        return ""

    try:
        from src.utils.audio_utils import create_xwm, extract_bsa, extract_fuz

        os.makedirs(os.path.join(get_app_root(), "temp"), exist_ok=True)
        extract_bsa(ref)

        fuz_path = os.path.join(get_app_root(), "temp", f"{stem}.fuz")
        xwm_path = os.path.join(get_app_root(), "temp", f"{stem}.xwm")

        extract_fuz(fuz_path)
        create_xwm(xwm_path, wav_path, False)

        for path in (
            xwm_path,
            fuz_path,
            os.path.join(get_app_root(), "temp", f"{stem}.lip"),
        ):
            if os.path.exists(path):
                os.remove(path)
    except Exception:
        return ""

    return wav_path if os.path.exists(wav_path) else ""


class ReferencesPage(Page):
    page_id = "references"
    label = "Reference Audio"
    icon = "fa-microphone"
    nav_group = 0
    nav_position = "top"

    def __init__(self, state: AppState):
        self._state = state
        self._ref_filter = [""]
        self._extracting: set[int] = set()
        self._show_selected_only = [False]
        self._active_tab = "Fallout 4"
        self._focused_reference: int | None = None
        self._duration_cache: dict[str, str] = {}

    def _select_reference(self, idx: int, ref):
        self._focused_reference = idx
        self._state.selected_references.add(idx)
        path = _resolve_reference_path(ref)
        if path:
            try:
                import soundfile as sf

                info = sf.info(path)
                self._state.reference_audio_length += max(0.0, float(info.duration))
            except Exception:
                pass

    def _deselect_reference(self, idx: int, ref):
        self._focused_reference = idx
        self._state.selected_references.discard(idx)
        path = _resolve_reference_path(ref)
        if path:
            try:
                import soundfile as sf

                info = sf.info(path)
                self._state.reference_audio_length = max(
                    0.0,
                    self._state.reference_audio_length - float(info.duration),
                )
            except Exception:
                pass

    def _play_reference(self, idx: int, ref):
        logger.info("play_reference idx=%s ref=%r", idx, ref)
        try:
            path = _resolve_reference_path(ref)
            if path:
                self._focused_reference = idx
                self._state.current_audio_file = path
                self._state.play_audio_requested = True
        except Exception as exc:
            logger.exception("play_reference failed: %s", exc)
        finally:
            self._extracting.discard(idx)

    def _clear_selection(self):
        self._state.selected_references.clear()
        self._state.reference_audio_length = 0.0

    def _select_filtered(self, custom_only: bool):
        for idx, ref in self._iter_filtered_references(custom_only):
            if idx not in self._state.selected_references:
                self._select_reference(idx, ref)

    def _iter_filtered_references(self, custom_only: bool):
        for idx, ref in enumerate(self._state.reference_audio):
            if self._is_custom_reference(ref) != custom_only:
                continue
            if self._show_selected_only[0] and idx not in self._state.selected_references:
                continue
            if not self._matches_filter(ref):
                continue
            yield idx, ref

    @staticmethod
    def _is_custom_reference(ref) -> bool:
        return isinstance(ref, str) or (isinstance(ref, dict) and bool(ref.get("path")))

    def _matches_filter(self, ref) -> bool:
        text = self._ref_filter[0].strip().lower()
        if not text:
            return True
        filename, dialogue, plugin, folder, _duration = self._reference_fields(ref)
        haystack = " ".join(part for part in (filename, dialogue, plugin, folder) if part).lower()
        return text in haystack

    def _reference_fields(self, ref):
        if isinstance(ref, str):
            return (
                os.path.basename(ref),
                "",
                "Custom",
                os.path.dirname(ref),
                self._get_duration_text(ref),
            )
        if isinstance(ref, dict):
            return (
                ref.get("filename", os.path.basename(ref.get("path", ""))),
                ref.get("dialogue", ""),
                ref.get("plugin", ""),
                ref.get("folder", ""),
                f"{ref.get('duration', 0):.1f}s" if ref.get("duration") else "",
            )
        return "", "", "", "", ""

    def _get_duration_text(self, path: str) -> str:
        if path in self._duration_cache:
            return self._duration_cache[path]
        duration_text = ""
        try:
            import soundfile as sf

            info = sf.info(path)
            if info.duration:
                duration_text = f"{float(info.duration):.1f}s"
        except Exception:
            duration_text = ""
        self._duration_cache[path] = duration_text
        return duration_text

    def _focused_entry(self):
        idx = self._focused_reference
        if idx is None or idx < 0 or idx >= len(self._state.reference_audio):
            self._focused_reference = None
            return None
        return idx, self._state.reference_audio[idx]

    def _draw_summary(self):
        from imgui_bundle import imgui

        selected_count = len(self._state.selected_references)
        total_refs = len(self._state.reference_audio)
        total_dur = self._state.reference_audio_length
        min_ref = self._state.engine_type.min_reference_length if self._state.engine_type else 0
        max_ref = self._state.engine_type.max_reference_length if self._state.engine_type else 0

        status_color = (0.75, 0.75, 0.75, 1.0)
        status_text = "No reference selected. The model default will be used."
        if selected_count:
            if min_ref and total_dur < min_ref:
                status_color = (0.96, 0.75, 0.33, 1.0)
                status_text = f"Selected references are short. Target {min_ref:.0f}-{max_ref:.0f}s."
            elif max_ref and total_dur > max_ref:
                status_color = (0.95, 0.42, 0.34, 1.0)
                status_text = f"Selected references exceed the recommended range. Target {min_ref:.0f}-{max_ref:.0f}s."
            else:
                status_color = (0.48, 0.84, 0.54, 1.0)
                engine_label = self._state.engine_type.value if self._state.engine_type else "the current engine"
                status_text = f"Reference duration is in range for {engine_label}."

        imgui.text_disabled("Reference Summary")
        imgui.separator()
        imgui.text(f"Selected: {selected_count}/{total_refs}")
        imgui.same_line()
        imgui.text(f"Total Duration: {total_dur:.1f}s")
        imgui.text_colored(status_color, status_text)

    def _draw_focus_panel(self):
        from imgui_bundle import imgui

        entry = self._focused_entry()
        imgui.text_disabled("Focused Reference")
        imgui.separator()

        if not entry:
            imgui.text_wrapped(
                "Pick a row from Fallout 4 or Custom references to preview it, then use Select or Remove here."
            )
            return

        idx, ref = entry
        filename, dialogue, plugin, folder, duration = self._reference_fields(ref)
        is_selected = idx in self._state.selected_references

        imgui.text(filename or "Unnamed Reference")
        source = plugin or folder or "Custom Reference"
        if duration:
            imgui.same_line()
            imgui.text_disabled(duration)
        imgui.text_disabled(source)
        if dialogue:
            imgui.text_wrapped(dialogue)

        if is_selected:
            if imgui.button("Remove Focused Reference##ref_focus_remove"):
                self._deselect_reference(idx, ref)
        else:
            push_accent_button_style()
            try:
                if imgui.button("Select Focused Reference##ref_focus_select"):
                    self._select_reference(idx, ref)
            finally:
                pop_accent_button_style()

        imgui.same_line()
        if idx in self._extracting:
            imgui.text_disabled("Resolving preview...")
        elif imgui.button("Preview##ref_focus_preview"):
            self._extracting.add(idx)
            threading.Thread(target=self._play_reference, args=(idx, ref), daemon=True).start()

        if self._is_custom_reference(ref):
            parent_dir = os.path.dirname(_resolve_reference_path(ref) or "")
            if parent_dir and hasattr(os, "startfile"):
                imgui.same_line()
                if imgui.button("Open Folder##ref_focus_folder"):
                    os.startfile(parent_dir)

    def _draw_toolbar(self, custom_only: bool):
        from imgui_bundle import imgui
        from src.ui_imgui.widgets.tables import search_filter

        search_filter("ref_search", self._ref_filter, width=320.0)
        imgui.same_line()
        if imgui.button("Select Visible##ref_visible"):
            self._select_filtered(custom_only)
        imgui.same_line()
        if imgui.button("Clear Selection##ref_clear"):
            self._clear_selection()
        imgui.same_line()
        _, self._show_selected_only[0] = imgui.checkbox("Show Highlighted", self._show_selected_only[0])

    def _draw_reference_table(self, table_id: str, refs: list[tuple[int, object]]):
        from imgui_bundle import imgui
        from src.ui_imgui.widgets.tables import begin_table, end_table

        cols = ["Action", "Preview", "Filename", "Dialogue", "Plugin", "Folder", "Duration"]
        if begin_table(f"ref_table_{table_id}", cols):
            accent_row = imgui.get_color_u32(imgui.ImVec4(*get_accent_color(0.18)))
            focused_row = imgui.get_color_u32(imgui.ImVec4(*get_accent_color(0.08)))
            for idx, ref in refs:
                filename, dialogue, plugin, folder, duration = self._reference_fields(ref)
                if not filename:
                    continue

                imgui.table_next_row()
                is_selected = idx in self._state.selected_references
                is_focused = idx == self._focused_reference
                if is_selected:
                    imgui.table_set_bg_color(imgui.TableBgTarget_.row_bg0, accent_row)
                elif is_focused:
                    imgui.table_set_bg_color(imgui.TableBgTarget_.row_bg0, focused_row)

                imgui.table_set_column_index(0)
                if is_selected:
                    if imgui.small_button(f"Remove##ref_remove_{idx}"):
                        self._deselect_reference(idx, ref)
                else:
                    if imgui.small_button(f"Select##ref_select_{idx}"):
                        self._select_reference(idx, ref)

                imgui.table_set_column_index(1)
                if idx in self._extracting:
                    imgui.text_disabled("...")
                elif imgui.small_button(f"Play##ref_play_{idx}"):
                    self._focused_reference = idx
                    self._extracting.add(idx)
                    threading.Thread(target=self._play_reference, args=(idx, ref), daemon=True).start()

                imgui.table_set_column_index(2)
                if imgui.small_button(f"{filename}##ref_focus_{idx}"):
                    self._focused_reference = idx
                if imgui.is_item_hovered() and filename:
                    imgui.set_tooltip(filename)

                imgui.table_set_column_index(3)
                if dialogue:
                    if is_selected:
                        imgui.text_colored(imgui.ImVec4(*get_accent_color()), dialogue)
                    else:
                        imgui.text(dialogue)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(dialogue)
                else:
                    imgui.text_disabled("Custom reference")

                imgui.table_set_column_index(4)
                imgui.text(plugin or "Custom")

                imgui.table_set_column_index(5)
                imgui.text(folder or "")
                if folder and imgui.is_item_hovered():
                    imgui.set_tooltip(folder)

                imgui.table_set_column_index(6)
                if duration:
                    imgui.text(duration)
                else:
                    imgui.text_disabled("--")
            end_table()

    def draw(self):
        from imgui_bundle import imgui

        if self._state.engine_type == EngineType.RVC:
            imgui.text_disabled("Reference Audio")
            imgui.separator()
            imgui.text_wrapped(
                "Reference audio selection is disabled while the active engine is RVC. "
                "RVC clones direct input instead of using the shared reference workflow."
            )
            return

        fallout_refs = list(self._iter_filtered_references(custom_only=False))
        custom_refs = list(self._iter_filtered_references(custom_only=True))

        self._draw_summary()
        imgui.spacing()
        self._draw_focus_panel()
        imgui.spacing()

        current_tab_is_custom = self._active_tab == "Custom"
        self._draw_toolbar(current_tab_is_custom)
        imgui.separator()

        if imgui.begin_tab_bar("reference_tabs"):
            if imgui.begin_tab_item(f"Fallout 4 ({len(fallout_refs)})")[0]:
                self._active_tab = "Fallout 4"
                if fallout_refs:
                    self._draw_reference_table("fallout4", fallout_refs)
                else:
                    imgui.text_disabled("No Fallout 4 references are available for the current character.")
                imgui.end_tab_item()

            if imgui.begin_tab_item(f"Custom ({len(custom_refs)})")[0]:
                self._active_tab = "Custom"
                if custom_refs:
                    self._draw_reference_table("custom", custom_refs)
                else:
                    imgui.text_disabled("No files were found in the references folder.")
                imgui.end_tab_item()

            imgui.end_tab_bar()
