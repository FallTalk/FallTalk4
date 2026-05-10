from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState


class Toolbar:
    """
    Horizontal toolbar at the top of the window, rendered via hello_imgui edge toolbar.
    Shows current page label, character info, engine/device dropdowns, reset button.
    """
    SIZE_EM = 4.0

    def __init__(self, state: AppState):
        self._state = state
        self._page_label = ""

    def set_page_label(self, label: str):
        self._page_label = label

    def draw(self):
        from imgui_bundle import imgui, hello_imgui, icons_fontawesome_6 as fa
        from src.config.config import cfg
        from src.enums.engine_type import EngineType

        page_label = self._page_label or "FallTalk"
        char_name = (
            self._state.pending_character
            or (self._state.tts_engine.model_name if self._state.tts_engine and self._state.tts_engine.model_name else None)
            or "No model loaded"
        )
        ref_count = len(self._state.reference_audio)
        selected_ref_count = len(self._state.selected_references)
        ref_dur = self._state.reference_audio_length
        ref_summary = (
            f"{selected_ref_count}/{ref_count} selected ({ref_dur:.1f}s)"
            if ref_count
            else "No references selected"
        )

        if imgui.button(f"{fa.ICON_FA_BARS}##toggle_nav"):
            self._state.nav_collapsed = not self._state.nav_collapsed
            cfg.set(cfg.nav_collapsed, self._state.nav_collapsed)
        imgui.same_line()

        imgui.align_text_to_frame_padding()
        imgui.text(page_label)
        imgui.same_line()
        imgui.text_disabled("|")
        imgui.same_line()
        imgui.text_disabled("Character")
        imgui.same_line()
        imgui.text(char_name)
        imgui.same_line()
        imgui.text_disabled("|")
        imgui.same_line()
        imgui.text_disabled("References")
        imgui.same_line()
        imgui.text(ref_summary)

        imgui.spacing()

        # Engine dropdown
        engines = [e.value for e in EngineType if e.enabled]
        current_engine = cfg.get(cfg.engine)
        idx = engines.index(current_engine) if current_engine in engines else 0
        imgui.align_text_to_frame_padding()
        imgui.text_disabled("Engine")
        imgui.same_line()
        imgui.set_next_item_width(hello_imgui.em_size(10.0))
        changed, new_idx = imgui.combo("##tb_engine", idx, engines)
        if changed:
            cfg.set(cfg.engine, engines[new_idx])
            self._state.engine_type = EngineType(engines[new_idx])
        imgui.same_line()

        # Device dropdown
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
        imgui.align_text_to_frame_padding()
        imgui.text_disabled("Device")
        imgui.same_line()
        imgui.set_next_item_width(hello_imgui.em_size(5.0))
        changed, new_dev = imgui.combo("##tb_device", dev_idx, devices)
        if changed:
            cfg.set(cfg.device, devices[new_dev])
        imgui.same_line()

        # Reset button
        if imgui.button("Reset All Settings##tb"):
            nav_collapsed = self._state.nav_collapsed
            cfg.reset()
            cfg.set(cfg.nav_collapsed, nav_collapsed)
            self._state.nav_collapsed = nav_collapsed
