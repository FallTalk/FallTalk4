from __future__ import annotations

import os
import webbrowser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState

from src.ui_imgui.page import Page
from src.config.config import (
    cfg,
    HELP_URL,
    NEXUS_URL,
    KOFI_URL,
    DISCORD_URL,
    HUGGING_FACE,
    VERSION,
    AUTHOR,
    YEAR,
)
from src.enums.engine_type import EngineType
from src.ui_imgui.widgets.common import hex_to_rgba, rgba_to_hex


class SettingsPage(Page):
    page_id = "settings"
    label = "Settings"
    icon = "fa-gear"
    nav_group = 3
    nav_position = "bottom"
    show_audio = False

    def __init__(self, state: AppState):
        self._state = state
        self._theme_hex = [cfg.get(cfg.theme_color)]
        self._theme_presets = [
            ("New Vegas", "#ffb642"),
            ("Fallout 76", "#f5cb5b"),
            ("Fallout 4", "#0b89d5"),
            ("Fallout 3", "#146c11"),
            ("Dracula", "#bd93f9"),
        ]

    def draw(self):
        from imgui_bundle import imgui
        from src.ui_imgui.widgets.engine_settings import draw_rvc_settings_panel

        imgui.text("Global Settings")
        imgui.separator()

        if imgui.begin_table(
            "settings_layout",
            2,
            imgui.TableFlags_.borders_inner_v | imgui.TableFlags_.sizing_stretch_same | imgui.TableFlags_.resizable,
        ):
            imgui.table_next_row()
            imgui.table_set_column_index(0)
            self._draw_general_section(imgui)
            imgui.spacing()
            self._draw_paths_section(imgui)
            imgui.spacing()
            self._draw_audio_section(imgui)
            imgui.spacing()
            self._draw_bulk_section(imgui)

            imgui.table_set_column_index(1)
            self._draw_appearance_section(imgui)
            imgui.spacing()
            self._draw_hugging_face_section(imgui)
            imgui.spacing()
            self._draw_text_processing_section(imgui)
            imgui.spacing()
            self._draw_features_section(imgui)
            imgui.spacing()
            _section_title("Advanced RVC")
            draw_rvc_settings_panel()
            imgui.spacing()
            self._draw_about_section(imgui)
            imgui.end_table()

        imgui.spacing()
        imgui.separator()
        if imgui.button("Reset All to Defaults", size=(200, 0)):
            nav_collapsed = self._state.nav_collapsed
            cfg.reset()
            cfg.set(cfg.nav_collapsed, nav_collapsed)
            self._state.nav_collapsed = nav_collapsed

    def _draw_general_section(self, imgui):
        _section_title("General")

        engines = [e.value for e in EngineType if e.enabled]
        current = cfg.get(cfg.engine)
        idx = engines.index(current) if current in engines else 0
        imgui.text("Engine")
        imgui.set_next_item_width(-1)
        changed, new_idx = imgui.combo("##engine", idx, engines)
        if changed:
            cfg.set(cfg.engine, engines[new_idx])
            self._state.engine_type = EngineType(engines[new_idx])

        devices = ["cuda", "cpu"]
        try:
            import torch
            if torch.cuda.device_count() > 1:
                for i in range(torch.cuda.device_count()):
                    devices.append(f"cuda:{i}")
        except ImportError:
            pass

        cur_device = cfg.get(cfg.device)
        dev_idx = devices.index(cur_device) if cur_device in devices else 0
        imgui.text("Device")
        imgui.set_next_item_width(-1)
        changed, new_dev = imgui.combo("##device", dev_idx, devices)
        if changed:
            cfg.set(cfg.device, devices[new_dev])

        _checkbox("Auto-Play Audio", cfg.auto_play)
        _checkbox("Check for Updates", cfg.check_for_updates)
        _checkbox("Auto Download Configs", cfg.download_configs)
        _checkbox("Load Engine at Start", cfg.load_engine_art_start)
        _int_slider("Seed (-1 = random)##seed", cfg.seed, -1, 999999)

    def _draw_paths_section(self, imgui):
        _section_title("Paths")
        _path_picker("Output Directory", cfg.output_dir)
        _path_picker("Custom References", cfg.custom_references)
        _path_picker("Fallout 4 Directory", cfg.fallout_4_directory)
        _path_picker("HuggingFace Cache", cfg.huggingface_cache_dir)

    def _draw_audio_section(self, imgui):
        _section_title("Audio Devices")
        try:
            import sounddevice as sd
            devices_list = sd.query_devices()
            out_names = [d["name"] for d in devices_list if d["max_output_channels"] > 0]
            in_names = [d["name"] for d in devices_list if d["max_input_channels"] > 0]
        except Exception:
            out_names = ["Default"]
            in_names = ["Default"]

        out_idx = cfg.get(cfg.audio_output_device)
        if out_idx < 0 or out_idx >= len(out_names):
            out_idx = 0
        imgui.text("Output Device")
        imgui.set_next_item_width(-1)
        changed, new_idx = imgui.combo("##audio_out", out_idx, out_names)
        if changed:
            cfg.set(cfg.audio_output_device, new_idx)

        in_idx = cfg.get(cfg.audio_input_device)
        if in_idx < 0 or in_idx >= len(in_names):
            in_idx = 0
        imgui.text("Input Device")
        imgui.set_next_item_width(-1)
        changed, new_idx = imgui.combo("##audio_in", in_idx, in_names)
        if changed:
            cfg.set(cfg.audio_input_device, new_idx)

    def _draw_bulk_section(self, imgui):
        _section_title("Bulk & Workflow")
        _checkbox("Replace Existing##bulk", cfg.replace_existing)
        _checkbox("Include Subdirectories##bulk", cfg.include_subdir)
        _int_slider("Threads##bulk", cfg.threads, 1, 16)
        _int_slider("Multi-Gen Count##bulk", cfg.multigen_total, 1, 20)
        _int_slider("EzVoice Count##bulk", cfg.ez_total, 1, 20)

    def _draw_appearance_section(self, imgui):
        _section_title("Appearance")
        changed, val = imgui.slider_float(
            "UI Scale (restart required)##dpi", cfg.get(cfg.font_global_scale), 0.5, 3.0, format="%.1f"
        )
        if changed:
            cfg.set(cfg.font_global_scale, val)
        imgui.text_disabled("Saved to config. Restart FallTalk to apply font scaling.")

        current_hex = cfg.get(cfg.theme_color)
        if self._theme_hex[0] != current_hex:
            self._theme_hex[0] = current_hex

        imgui.text("Accent Color")
        imgui.set_next_item_width(180)
        changed, new_hex = imgui.input_text("##theme_hex", self._theme_hex[0])
        if changed:
            self._theme_hex[0] = new_hex
            cleaned = new_hex.strip()
            if len(cleaned.lstrip("#")) == 6:
                cfg.set(cfg.theme_color, f"#{cleaned.lstrip('#')}")

        imgui.same_line()
        preview = hex_to_rgba(cfg.get(cfg.theme_color))
        changed, color = imgui.color_edit3("##theme_color", (preview[0], preview[1], preview[2]))
        if changed:
            new_color = rgba_to_hex(color)
            self._theme_hex[0] = new_color
            cfg.set(cfg.theme_color, new_color)

        imgui.text_disabled("Presets")
        for idx, (label, value) in enumerate(self._theme_presets):
            imgui.push_style_color(imgui.Col_.button, hex_to_rgba(value))
            imgui.push_style_color(imgui.Col_.button_hovered, hex_to_rgba(value))
            imgui.push_style_color(imgui.Col_.button_active, hex_to_rgba(value))
            if imgui.button(f"{label}##theme_{idx}"):
                self._theme_hex[0] = value
                cfg.set(cfg.theme_color, value)
            imgui.pop_style_color(3)
            if idx < len(self._theme_presets) - 1:
                imgui.same_line()

        changed, collapsed = imgui.checkbox("Collapse Navigation by Default", cfg.get(cfg.nav_collapsed))
        if changed:
            cfg.set(cfg.nav_collapsed, collapsed)
            self._state.nav_collapsed = collapsed

    def _draw_hugging_face_section(self, imgui):
        _section_title("Hugging Face")
        imgui.text("API Access Token")
        imgui.set_next_item_width(-1)
        changed, token = imgui.input_text("##hf_token", cfg.get(cfg.huggingface_key))
        if changed:
            cfg.set(cfg.huggingface_key, token)
        _checkbox("Disable SSL Verify", cfg.disableSSLVerify)
        if imgui.button("Open Hugging Face##settings"):
            webbrowser.open(HUGGING_FACE)
        imgui.same_line()
        if imgui.button("Open GitHub##settings"):
            webbrowser.open(HELP_URL)

    def _draw_text_processing_section(self, imgui):
        _section_title("Text Processing")
        _int_slider("Max Characters##tp", cfg.max_text_size, 10, 2000)
        _int_slider("Min Chunk Size##tp", cfg.min_chunk_size, 1, 500)
        _checkbox("Pad Short Phrases##tp", cfg.pad_short_phrases)
        _checkbox("Lowercase Conversion##tp", cfg.lowercase_conversion)
        _checkbox("Whitespace Normalization##tp", cfg.whitespace_normalization)
        _checkbox("Dot-Letter Fix##tp", cfg.dot_letter_fix)
        _checkbox("Inline Reference Removal##tp", cfg.inline_reference_removal)

    def _draw_features_section(self, imgui):
        _section_title("Features")
        _checkbox("RVC Enabled", cfg.rvc_enabled)
        _checkbox("Audio Enhancement", cfg.apbwe_enabled)
        _checkbox("Create FUZ", cfg.xwm_enabled)
        _checkbox("Keep Only FUZ", cfg.keep_only_fuz)
        _checkbox("Use Existing LIP", cfg.use_existing_lip)
        _checkbox("API Only Mode (headless)", cfg.api_only_mode)

    def _draw_about_section(self, imgui):
        _section_title("About")
        imgui.text(f"FallTalk {VERSION}")
        imgui.text_disabled(f"Copyright {YEAR}, {AUTHOR}")
        if imgui.button("GitHub##about"):
            webbrowser.open(HELP_URL)
        imgui.same_line()
        if imgui.button("Nexus##about"):
            webbrowser.open(NEXUS_URL)
        imgui.same_line()
        if imgui.button("Discord##about"):
            webbrowser.open(DISCORD_URL)
        imgui.same_line()
        if imgui.button("Ko-fi##about"):
            webbrowser.open(KOFI_URL)


def _checkbox(label: str, config_key):
    from imgui_bundle import imgui
    changed, val = imgui.checkbox(label, cfg.get(config_key))
    if changed:
        cfg.set(config_key, val)


def _int_slider(label: str, config_key, lo: int, hi: int):
    from imgui_bundle import imgui
    changed, val = imgui.slider_int(label, cfg.get(config_key), lo, hi)
    if changed:
        cfg.set(config_key, val)


def _path_picker(label: str, config_key):
    from imgui_bundle import imgui
    current = cfg.get(config_key) or ""
    imgui.text(label)
    imgui.set_next_item_width(320)
    changed, new_val = imgui.input_text(f"##{label}", current)
    if changed:
        cfg.set(config_key, new_val)
        if config_key == cfg.huggingface_cache_dir and new_val:
            os.environ["HF_HUB_CACHE"] = new_val
    imgui.same_line()
    if imgui.button(f"Browse##{label}"):
        from src.ui_imgui.widgets.file_dialog import open_folder
        folder = open_folder(f"Select {label}")
        if folder:
            cfg.set(config_key, folder)
            if config_key == cfg.huggingface_cache_dir:
                os.environ["HF_HUB_CACHE"] = folder


def _section_title(label: str):
    from imgui_bundle import imgui
    imgui.text_disabled(label)
    imgui.separator()
