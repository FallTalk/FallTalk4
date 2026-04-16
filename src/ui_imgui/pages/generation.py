from __future__ import annotations

import os
import shutil
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState

from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.ui_imgui.page import Page
from src.ui_imgui.widgets.drawer import Drawer
from src.ui_imgui.widgets.recorder_panel import RecorderPanel
from src.ui_imgui.widgets.common import (
    draw_page_action_strip,
    pop_accent_button_style,
    push_accent_button_style,
)


class GenerationPage(Page):
    page_id = "generation"
    label = "Generation"
    icon = "fa-play"
    nav_group = 0
    nav_position = "top"

    _RVC_TABS = [
        ("Microphone", "Microphone"),
        ("File", "File"),
        ("Edge TTS", "EdgeTTS"),
        ("ElevenLabs", "Eleven-Labs"),
    ]

    def __init__(self, state: AppState):
        self._state = state
        self._drawer = Drawer()
        self._recorder_panel = RecorderPanel(state)
        self._rvc_file_path = [""]
        self._edge_text = [""]
        self._eleven_text = [""]
        self._edge_voices: list[str] = []
        self._eleven_voices: list[str] = []
        self._edge_voice_idx = [0]
        self._eleven_voice_idx = [0]
        self._edge_loading = False
        self._eleven_loading = False
        self._edge_error = ""
        self._eleven_error = ""

    def draw(self):
        from imgui_bundle import icons_fontawesome_6 as fa, imgui

        if self._state.recording_complete and self._state.engine_type == EngineType.RVC:
            self._state.recording_complete = False
            self._start_rvc_from_recording()

        draw_page_action_strip(
            "gen_settings",
            "gen_help",
            "Engine Settings",
            "Engine Help",
            lambda: self._open_settings(fa.ICON_FA_GEAR),
            lambda: self._open_help(fa.ICON_FA_CIRCLE_QUESTION),
        )

        if self._state.engine_type == EngineType.RVC:
            self._draw_rvc_page()
        else:
            self._draw_standard_generation_page()

        self._drawer.draw()

    def _draw_standard_generation_page(self):
        from imgui_bundle import imgui

        engine_loaded = self._state.tts_engine is not None
        selected_refs = len(self._state.selected_references)
        ref_duration = self._state.reference_audio_length
        engine_name = self._state.engine_type.value if self._state.engine_type else "No Engine"

        imgui.begin_child(
            "##generation_overview",
            size=imgui.ImVec2(0, 72),
            child_flags=imgui.ChildFlags_.borders,
        )
        try:
            imgui.text_disabled("Generation Overview")
            imgui.separator()
            imgui.text(f"Engine: {engine_name}")
            imgui.same_line()
            imgui.text(f"Selected References: {selected_refs}")
            imgui.same_line()
            imgui.text(f"Reference Duration: {ref_duration:.1f}s")
        finally:
            imgui.end_child()

        imgui.spacing()
        imgui.text_disabled("Input")
        imgui.separator()
        imgui.text("Text to Generate")
        imgui.set_next_item_width(-1)
        _, self._state.text_input = imgui.input_text_multiline(
            "##text_input", self._state.text_input, size=(0, 170)
        )

        imgui.spacing()
        imgui.text_disabled("Primary Controls")
        imgui.separator()
        imgui.set_next_item_width(320)
        _, self._state.output_name = imgui.input_text("Output Name##out_name", self._state.output_name)
        changed, auto_play = imgui.checkbox("Auto-Play Result", cfg.get(cfg.auto_play))
        if changed:
            cfg.set(cfg.auto_play, auto_play)

        changed, val = imgui.checkbox("RVC", cfg.get(cfg.rvc_enabled))
        if changed:
            cfg.set(cfg.rvc_enabled, val)
        imgui.same_line()
        changed, val = imgui.checkbox("Super Resolution", cfg.get(cfg.apbwe_enabled))
        if changed:
            cfg.set(cfg.apbwe_enabled, val)

        imgui.spacing()
        from src.ui_imgui.widgets.engine_settings import draw_engine_quick_tuning

        draw_engine_quick_tuning(self._state)
        imgui.spacing()
        imgui.separator()

        if not engine_loaded:
            imgui.text_colored(
                (0.7, 0.7, 0.7, 1.0),
                "Load a character model from Character Models before generating audio.",
            )
        else:
            push_accent_button_style()
            try:
                if imgui.button("Generate Audio", size=(170, 0)):
                    self._start_generation()
            finally:
                pop_accent_button_style()

    def _draw_rvc_page(self):
        from imgui_bundle import imgui

        engine_loaded = self._state.tts_engine is not None and self._state.engine_type == EngineType.RVC
        model_name = self._state.tts_engine.model_name if engine_loaded and self._state.tts_engine else "No Model Loaded"
        current_mode = cfg.get(cfg.rvc_mode) or "Microphone"
        current_mode_label = next((label for label, value in self._RVC_TABS if value == current_mode), current_mode)

        imgui.text_disabled("RVC")
        imgui.separator()
        imgui.text(f"Loaded Character: {model_name}")
        imgui.same_line()
        imgui.text_disabled(f"Mode: {current_mode_label}")
        if not engine_loaded:
            imgui.text_colored(
                (0.96, 0.75, 0.33, 1.0),
                "Load an RVC-enabled character model from Character Models before converting audio.",
            )
            imgui.separator()

        if imgui.begin_tab_bar("rvc_mode_tabs"):
            try:
                for label, mode_value in self._RVC_TABS:
                    tab_open, _ = imgui.begin_tab_item(label)
                    if not tab_open:
                        continue
                    try:
                        if cfg.get(cfg.rvc_mode) != mode_value:
                            cfg.set(cfg.rvc_mode, mode_value)
                        if mode_value == "Microphone":
                            self._draw_rvc_microphone_tab()
                        elif mode_value == "File":
                            self._draw_rvc_file_tab()
                        elif mode_value == "EdgeTTS":
                            self._draw_rvc_edge_tts_tab()
                        elif mode_value == "Eleven-Labs":
                            self._draw_rvc_eleven_labs_tab()
                    finally:
                        imgui.end_tab_item()
            finally:
                imgui.end_tab_bar()

    def _draw_rvc_common_controls(self):
        from imgui_bundle import imgui

        imgui.separator()
        imgui.text_disabled("Output")
        imgui.set_next_item_width(320)
        _, self._state.output_name = imgui.input_text("Output Name##rvc_out_name", self._state.output_name)
        changed, auto_play = imgui.checkbox("Auto-Play Result##rvc", cfg.get(cfg.auto_play))
        if changed:
            cfg.set(cfg.auto_play, auto_play)

    def _draw_rvc_microphone_tab(self):
        from imgui_bundle import imgui

        imgui.text_disabled("Microphone")
        imgui.separator()
        imgui.text_wrapped("Record a line, stop the recorder, and the result will be cloned through the loaded RVC voice.")
        self._recorder_panel.draw()
        self._draw_rvc_common_controls()

    def _draw_rvc_file_tab(self):
        from imgui_bundle import imgui
        from src.ui_imgui.widgets.file_dialog import open_file

        imgui.text_disabled("File Input")
        imgui.separator()
        imgui.text("Audio File")
        imgui.set_next_item_width(420)
        _, self._rvc_file_path[0] = imgui.input_text("##rvc_file_path", self._rvc_file_path[0])
        imgui.same_line()
        if imgui.button("Browse##rvc_file"):
            selected = open_file("Select Audio File", [("Audio Files", "*.wav *.mp3"), ("WAV Files", "*.wav"), ("MP3 Files", "*.mp3")])
            if selected:
                self._rvc_file_path[0] = selected

        changed, val = imgui.slider_int("Pitch Adjustment##rvc_file", cfg.get(cfg.rvc_pitch), -24, 24)
        if changed:
            cfg.set(cfg.rvc_pitch, val)

        self._draw_rvc_common_controls()
        imgui.separator()
        push_accent_button_style()
        try:
            if imgui.button("Convert File", size=(170, 0)):
                self._start_rvc_from_file()
        finally:
            pop_accent_button_style()

    def _draw_rvc_edge_tts_tab(self):
        from imgui_bundle import imgui

        self._ensure_voice_fetch("edge")
        imgui.text_disabled("Edge TTS")
        imgui.separator()
        imgui.text_wrapped(
            "Edge TTS is the original free text source for RVC. Generate the base voice here, then clone it into the loaded character."
        )
        imgui.text("Text")
        _, self._edge_text[0] = imgui.input_text_multiline("##rvc_edge_text", self._edge_text[0], size=(0, 150))
        self._draw_voice_selector("edge")
        self._draw_rvc_common_controls()
        imgui.separator()
        push_accent_button_style()
        try:
            if imgui.button("Generate From Edge TTS", size=(210, 0)):
                self._start_edge_tts_generation()
        finally:
            pop_accent_button_style()

    def _draw_rvc_eleven_labs_tab(self):
        from imgui_bundle import imgui

        self._ensure_voice_fetch("eleven")
        imgui.text_disabled("ElevenLabs")
        imgui.separator()
        imgui.text_wrapped(
            "ElevenLabs can generate the source voice first, then FallTalk runs RVC over that result. "
            "Set your API key below if you want access to your ElevenLabs voices."
        )
        imgui.text("Text")
        _, self._eleven_text[0] = imgui.input_text_multiline("##rvc_eleven_text", self._eleven_text[0], size=(0, 130))
        current_key = cfg.get(cfg.rvc_eleven_labs_key) or ""
        changed, new_key = imgui.input_text(
            "API Access Key##rvc_eleven_key",
            current_key,
            flags=int(imgui.InputTextFlags_.password),
        )
        if changed:
            cfg.set(cfg.rvc_eleven_labs_key, new_key)
            self._eleven_voices.clear()
            self._eleven_voice_idx[0] = 0
            self._eleven_loading = False
            self._eleven_error = ""
        self._draw_voice_selector("eleven")
        self._draw_rvc_common_controls()
        imgui.separator()
        push_accent_button_style()
        try:
            if imgui.button("Generate From ElevenLabs", size=(220, 0)):
                self._start_eleven_labs_generation()
        finally:
            pop_accent_button_style()

    def _draw_voice_selector(self, provider: str):
        from imgui_bundle import imgui

        if provider == "edge":
            voices = self._edge_voices
            idx_ref = self._edge_voice_idx
            loading = self._edge_loading
            error = self._edge_error
            label = "Edge Voice"
            refresh_id = "edge"
        else:
            voices = self._eleven_voices
            idx_ref = self._eleven_voice_idx
            loading = self._eleven_loading
            error = self._eleven_error
            label = "ElevenLabs Voice"
            refresh_id = "eleven"

        imgui.text(label)
        if loading:
            imgui.same_line()
            imgui.text_disabled("Loading voices...")

        if error:
            imgui.text_colored((0.95, 0.42, 0.34, 1.0), error)

        if voices:
            idx_ref[0] = min(idx_ref[0], len(voices) - 1)
            imgui.set_next_item_width(420)
            _, idx_ref[0] = imgui.combo(f"##{provider}_voice_combo", idx_ref[0], voices)
        else:
            imgui.text_disabled("No voices loaded yet.")

        if imgui.button(f"Refresh Voices##{refresh_id}"):
            if provider == "edge":
                self._edge_voices.clear()
                self._edge_voice_idx[0] = 0
                self._edge_loading = False
                self._edge_error = ""
            else:
                self._eleven_voices.clear()
                self._eleven_voice_idx[0] = 0
                self._eleven_loading = False
                self._eleven_error = ""
            self._ensure_voice_fetch(provider, force=True)

    def _ensure_voice_fetch(self, provider: str, force: bool = False):
        if provider == "edge":
            if self._edge_loading or (not force and (self._edge_error or self._edge_voices)):
                return
            self._edge_loading = True
            self._edge_error = ""
            threading.Thread(target=self._load_edge_voices, daemon=True).start()
            return

        if self._eleven_loading or (not force and (self._eleven_error or self._eleven_voices)):
            return
        self._eleven_loading = True
        self._eleven_error = ""
        threading.Thread(target=self._load_eleven_voices, daemon=True).start()

    def _load_edge_voices(self):
        loop = None
        try:
            import asyncio
            from src.utils.inference_utils import get_edge_tts_voices

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            voices = get_edge_tts_voices()
            self._edge_voices = voices
            if voices:
                self._edge_voice_idx[0] = len(voices) - 1
        except Exception as exc:
            self._edge_error = str(exc)
        finally:
            try:
                if loop is not None:
                    loop.close()
            except Exception:
                pass
            try:
                import asyncio
                asyncio.set_event_loop(None)
            except Exception:
                pass
            self._edge_loading = False

    def _load_eleven_voices(self):
        try:
            from src.utils.inference_utils import get_eleven_labs_voices

            voices = get_eleven_labs_voices()
            self._eleven_voices = voices
            if voices:
                self._eleven_voice_idx[0] = len(voices) - 1
        except Exception as exc:
            self._eleven_error = str(exc)
        finally:
            self._eleven_loading = False

    def _open_settings(self, icon: str):
        from src.ui_imgui.widgets.engine_settings import draw_engine_settings

        self._drawer.open(lambda: draw_engine_settings(self._state), "Advanced Settings", icon)

    def _open_help(self, icon: str):
        from src.ui_imgui.widgets.engine_help import draw_engine_help

        self._drawer.open(lambda: draw_engine_help(self._state), "Engine Help", icon)

    def _require_loaded_engine(self) -> bool:
        if self._state.tts_engine is None or self._state.tts_engine.model_name is None:
            self._state.error_queue.put(("Error", "Please load a character model first."))
            return False
        return True

    def _build_output_file(self, engine_name: str, default_name: str | None = None) -> str:
        from src.utils.file_utils import get_output_file_name

        model_name = self._state.tts_engine.model_name if self._state.tts_engine else "rvc"
        return get_output_file_name(
            self._state.output_name or default_name,
            cfg.get(cfg.output_dir),
            model_name,
            engine_name,
        )

    def _start_generation(self):
        text = self._state.text_input.strip()
        if not text:
            self._state.error_queue.put(("Error", "Please enter text to generate."))
            return
        from src.ui_imgui.pages.references import _resolve_reference_path
        from src.utils.audio_utils import combine_references
        from src.utils.inference_utils import generic_inference, get_default_reference_and_transcript, preprocess_text

        selected = [
            self._state.reference_audio[i]
            for i in sorted(self._state.selected_references)
            if i < len(self._state.reference_audio)
        ]
        references = [path for path in (_resolve_reference_path(ref) for ref in selected) if path]

        transcribe_state = None
        dialogues = [ref.get("dialogue", "") for ref in selected if isinstance(ref, dict)]
        combined = " ".join(dialogue for dialogue in dialogues if dialogue)
        if combined:
            transcribe_state = {"transcript": combined}

        if not references and self._state.tts_engine:
            ref_path, transcribe_state = get_default_reference_and_transcript(
                self._state.callbacks,
                self._state.tts_engine.model_name,
            )
            if ref_path:
                references = [ref_path]
            else:
                self._state.error_queue.put(("Error", "Please select reference audio."))
                return

        output_file = self._build_output_file(self._state.engine_type.value, default_name=None)
        self._state.loading = True
        self._state.loading_message = "Generating audio..."
        threading.Thread(
            target=generic_inference,
            args=(
                self._state.callbacks,
                output_file,
                preprocess_text(text),
                combine_references(references),
                None,
                transcribe_state,
            ),
            kwargs={
                "speaker": self._state.tts_engine.model_name if self._state.tts_engine else None,
                "api": False,
            },
            daemon=True,
        ).start()

    def _start_rvc_from_recording(self):
        recording = self._state.recording_file
        if not recording:
            return
        if not self._require_loaded_engine():
            return
        from src.utils.inference_utils import rvc_inference

        output_file = self._build_output_file("RVC", default_name="rvc_recording")
        shutil.copy(recording, output_file)
        self._state.loading = True
        self._state.loading_message = "Cloning microphone audio..."
        threading.Thread(
            target=rvc_inference,
            args=(self._state.callbacks, output_file, None),
            daemon=True,
        ).start()

    def _start_rvc_from_file(self):
        source = self._rvc_file_path[0].strip()
        if not source:
            self._state.error_queue.put(("Error", "Please select an audio file to convert."))
            return
        if not os.path.exists(source):
            self._state.error_queue.put(("Error", "The selected audio file does not exist."))
            return
        if not self._require_loaded_engine():
            return
        from src.utils.inference_utils import rvc_inference

        output_file = self._build_output_file("RVC", default_name=os.path.splitext(os.path.basename(source))[0])
        shutil.copy(source, output_file)
        self._state.loading = True
        self._state.loading_message = "Cloning file audio..."
        threading.Thread(
            target=rvc_inference,
            args=(self._state.callbacks, output_file, None),
            daemon=True,
        ).start()

    def _start_edge_tts_generation(self):
        text = self._edge_text[0].strip()
        if not text:
            self._state.error_queue.put(("Error", "Please enter text for Edge TTS."))
            return
        if not self._edge_voices:
            self._state.error_queue.put(("Error", "Load Edge TTS voices before generating audio."))
            return
        if not self._require_loaded_engine():
            return
        from src.utils.inference_utils import edge_tts_inference, preprocess_text

        voice = self._edge_voices[self._edge_voice_idx[0]]
        output_file = self._build_output_file("RVC", default_name="edge_tts")
        self._state.loading = True
        self._state.loading_message = "Generating Edge TTS source..."
        threading.Thread(
            target=edge_tts_inference,
            args=(self._state.callbacks, preprocess_text(text), output_file, voice, None, False),
            daemon=True,
        ).start()

    def _start_eleven_labs_generation(self):
        text = self._eleven_text[0].strip()
        if not text:
            self._state.error_queue.put(("Error", "Please enter text for ElevenLabs."))
            return
        if not self._eleven_voices:
            self._state.error_queue.put(("Error", "Load ElevenLabs voices before generating audio."))
            return
        if not self._require_loaded_engine():
            return
        from src.utils.inference_utils import eleven_labs_inference, preprocess_text

        voice = self._eleven_voices[self._eleven_voice_idx[0]]
        output_file = self._build_output_file("RVC", default_name="elevenlabs")
        self._state.loading = True
        self._state.loading_message = "Calling ElevenLabs..."
        threading.Thread(
            target=eleven_labs_inference,
            args=(self._state.callbacks, preprocess_text(text), output_file, voice, None, False),
            daemon=True,
        ).start()
