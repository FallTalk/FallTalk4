from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState

from src.ui_imgui.page import Page
from src.config.config import cfg
from src.ui_imgui.widgets.common import push_accent_button_style, pop_accent_button_style, draw_page_action_strip
from src.ui_imgui.widgets.drawer import Drawer


class MultiGenerationPage(Page):
    page_id = "multi_generation"
    label = "Multi Generation"
    icon = "fa-list"
    nav_group = 0
    nav_position = "top"

    def __init__(self, state: AppState):
        self._state = state
        self._results: list[str] = []
        self._selected_result = [0]
        self._drawer = Drawer()

    def draw(self):
        from imgui_bundle import imgui, icons_fontawesome_6 as fa
        from src.ui_imgui.widgets.tables import begin_table, end_table

        engine_loaded = self._state.tts_engine is not None
        draw_page_action_strip(
            "multigen_settings",
            "multigen_help",
            "Engine Settings",
            "Engine Help",
            lambda: self._open_settings(fa.ICON_FA_GEAR),
            lambda: self._open_help(fa.ICON_FA_CIRCLE_QUESTION),
        )
        selected_count = len(self._state.selected_references)
        ref_secs = self._state.reference_audio_length
        min_ref = self._state.engine_type.min_reference_length if self._state.engine_type else 0
        max_ref = self._state.engine_type.max_reference_length if self._state.engine_type else 0
        needs_reference = bool(
            self._state.engine_type
            and (self._state.engine_type.needs_reference_when_trained or getattr(self._state.tts_engine, "is_base", False))
        )

        status_color = (0.75, 0.75, 0.75, 1.0)
        status_text = "Default reference will be used if none are selected."
        if selected_count:
            if min_ref and ref_secs < min_ref:
                status_color = (0.96, 0.75, 0.33, 1.0)
                status_text = f"Reference audio is short for this engine. Target {min_ref:.0f}-{max_ref:.0f}s."
            elif max_ref and ref_secs > max_ref:
                status_color = (0.95, 0.42, 0.34, 1.0)
                status_text = f"Reference audio is too long for this engine. Target {min_ref:.0f}-{max_ref:.0f}s."
            else:
                status_color = (0.48, 0.84, 0.54, 1.0)
                engine_label = self._state.engine_type.value if self._state.engine_type else "the current engine"
                status_text = f"Reference audio is in range for {engine_label}."
        elif needs_reference and self._state.tts_engine is not None:
            status_color = (0.82, 0.82, 0.62, 1.0)
            status_text = "No references selected. FallTalk will fall back to the model's default reference."

        imgui.text_disabled("Reference Summary")
        imgui.separator()
        imgui.text(f"Selected References: {selected_count}")
        imgui.same_line()
        imgui.text(f"Total Duration: {ref_secs:.1f}s")
        imgui.text_colored(status_color, status_text)

        multigen_total = cfg.get(cfg.multigen_total)
        if imgui.begin_table(
            "mg_layout",
            2,
            imgui.TableFlags_.borders_inner_v | imgui.TableFlags_.sizing_stretch_same | imgui.TableFlags_.resizable,
        ):
            imgui.table_next_row()
            imgui.table_set_column_index(0)
            imgui.text_disabled("Input")
            imgui.separator()
            imgui.text("Text to Generate")
            imgui.set_next_item_width(-1)
            _, self._state.text_input = imgui.input_text_multiline(
                "##mg_text_input", self._state.text_input, size=(0, 180)
            )

            imgui.table_set_column_index(1)
            imgui.text_disabled("Batch Settings")
            imgui.separator()
            imgui.text("Output Name Prefix")
            imgui.set_next_item_width(-1)
            _, self._state.output_name = imgui.input_text("##mg_out_name", self._state.output_name)
            changed, val = imgui.slider_int("Count##mg_count", multigen_total, 1, 20)
            if changed:
                cfg.set(cfg.multigen_total, val)
            changed, val = imgui.checkbox("RVC##mg", cfg.get(cfg.rvc_enabled))
            if changed:
                cfg.set(cfg.rvc_enabled, val)
            imgui.same_line()
            changed, val = imgui.checkbox("Super Resolution##mg", cfg.get(cfg.apbwe_enabled))
            if changed:
                cfg.set(cfg.apbwe_enabled, val)

            imgui.spacing()
            if not engine_loaded:
                imgui.text_colored((0.7, 0.7, 0.7, 1.0), "Load a character model before running a batch.")
            else:
                push_accent_button_style()
                clicked = imgui.button(f"Generate Batch ({multigen_total})", size=(220, 0))
                pop_accent_button_style()
                if clicked:
                    self._start_multigen()
            imgui.same_line()
            if imgui.button("Clear Results##mg", size=(120, 0)):
                self._results.clear()
                self._selected_result[0] = 0
            imgui.end_table()

        imgui.spacing()
        imgui.separator()

        imgui.text_disabled("Results")
        imgui.text(f"{len(self._results)} generated file(s)")
        if self._results:
            self._selected_result[0] = min(self._selected_result[0], len(self._results) - 1)
            current = self._results[self._selected_result[0]]
            imgui.text(f"Selected Result: {os.path.basename(current)}")
            imgui.same_line()
            if imgui.button("Play Selected##mg", size=(110, 0)):
                self._state.current_audio_file = current
                self._state.play_audio_requested = True
            imgui.same_line()
            if imgui.button("Save Selected##mg", size=(110, 0)):
                self._save_result(current)
        cols = ["Use", "File", "Preview", "Save"]
        if begin_table("##mg_results_table", cols):
            for i, path in enumerate(self._results):
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                label = "Selected" if self._selected_result[0] == i else "Use"
                if imgui.small_button(f"{label}##mg_sel_{i}"):
                    self._selected_result[0] = i
                imgui.table_set_column_index(1)
                imgui.text(os.path.basename(path))
                imgui.table_set_column_index(2)
                if imgui.small_button(f"Play##{i}"):
                    self._state.current_audio_file = path
                    self._state.play_audio_requested = True
                imgui.table_set_column_index(3)
                if imgui.small_button(f"Save##{i}"):
                    self._save_result(path)
            end_table()
        self._drawer.draw()

    def _start_multigen(self):
        text = self._state.text_input.strip()
        if not text:
            self._state.error_queue.put(("Error", "Please enter text to generate."))
            return
        if self._state.engine_type and self._state.selected_references:
            ref_length = self._state.reference_audio_length
            min_ref = self._state.engine_type.min_reference_length
            max_ref = self._state.engine_type.max_reference_length
            needs_reference = self._state.engine_type.needs_reference_when_trained or getattr(self._state.tts_engine, "is_base", False)
            if needs_reference and (ref_length < min_ref or ref_length > max_ref):
                self._state.error_queue.put((
                    "Reference Audio",
                    f"Please select between {min_ref} and {max_ref} seconds of reference audio for this engine.",
                ))
                return
        total = cfg.get(cfg.multigen_total)
        self._results.clear()
        self._selected_result[0] = 0
        self._state.loading = True
        self._state.loading_message = f"Multi-generating (0/{total})..."

        def _run():
            from src.utils.inference_utils import generic_inference, preprocess_text, get_default_reference_and_transcript
            from src.utils.audio_utils import combine_references
            from src.ui_imgui.pages.references import _resolve_reference_path
            from src.utils.file_utils import get_output_file_name
            selected = [self._state.reference_audio[i] for i in sorted(self._state.selected_references)
                         if i < len(self._state.reference_audio)]
            references = [p for p in (_resolve_reference_path(r) for r in selected) if p]
            dialogues = [r.get('dialogue', '') for r in selected if isinstance(r, dict)]
            combined = " ".join(d for d in dialogues if d)
            transcribe_state = {'transcript': combined} if combined else None
            if not references and self._state.tts_engine:
                ref_path, transcribe_state = get_default_reference_and_transcript(
                    self._state.callbacks, self._state.tts_engine.model_name)
                if ref_path:
                    references = [ref_path]
            for i in range(total):
                self._state.loading_message = f"Multi-generating ({i + 1}/{total})..."
                output_file = get_output_file_name(
                    f"{self._state.output_name}_{i + 1}" if self._state.output_name else None,
                    cfg.get(cfg.output_dir),
                    self._state.tts_engine.model_name if self._state.tts_engine else "unknown",
                    self._state.engine_type.value if self._state.engine_type else "unknown"
                )
                try:
                    generic_inference(
                        self._state.callbacks,
                        output_file,
                        preprocess_text(text),
                        combine_references(references),
                        None,
                        transcribe_state,
                        speaker=self._state.tts_engine.model_name if self._state.tts_engine else None,
                        api=False,
                    )
                    self._results.append(output_file)
                except Exception as e:
                    self._state.error_queue.put(("Error", str(e)))
            self._state.loading = False
            self._state.loading_message = ""

        threading.Thread(target=_run, daemon=True).start()

    def _open_settings(self, icon: str):
        from src.ui_imgui.widgets.engine_settings import draw_engine_settings
        self._drawer.open(lambda: draw_engine_settings(self._state), "Advanced Settings", icon)

    def _open_help(self, icon: str):
        from src.ui_imgui.widgets.engine_help import draw_engine_help
        self._drawer.open(lambda: draw_engine_help(self._state), "Engine Help", icon)

    def _save_result(self, path: str):
        import shutil
        from src.utils.file_utils import get_output_file_name
        dest = get_output_file_name(
            os.path.splitext(os.path.basename(path))[0],
            cfg.get(cfg.output_dir),
            "saved", "multigen"
        )
        shutil.copy2(path, dest)
