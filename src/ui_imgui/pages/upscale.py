from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState

from src.config.config import cfg
from src.ui_imgui.page import Page
from src.ui_imgui.widgets.drawer import Drawer
from src.ui_imgui.widgets.common import draw_page_action_strip


class UpscalePage(Page):
    page_id = "upscale"
    label = "Bulk Enhancement"
    icon = "fa-arrow-up"
    nav_group = 2
    nav_position = "top"
    show_audio = False

    def __init__(self, state: AppState):
        self._state = state
        self._drawer = Drawer()
        self._folder = [""]
        self._mode_idx = [1]
        self._sample_rate_idx = [0]
        self._modes = ["denoise", "isolate", "upscale"]
        self._sample_rates = ["44100", "48000"]

    def draw(self):
        from imgui_bundle import imgui, icons_fontawesome_6 as fa

        draw_page_action_strip(
            "upscale_settings",
            "upscale_help",
            "Bulk Enhancement Settings",
            "Bulk Enhancement Help",
            lambda: self._drawer.open(self._draw_settings_drawer, "Bulk Enhancement Settings", fa.ICON_FA_GEAR),
            lambda: self._drawer.open(self._draw_help_drawer, "Bulk Enhancement Help", fa.ICON_FA_CIRCLE_QUESTION),
        )

        imgui.text_disabled("Bulk Audio Enhancement")
        imgui.separator()

        imgui.text("Mode")
        for i, label in enumerate(["Denoise", "Isolate Vocals", "Upscale"]):
            if i > 0:
                imgui.same_line()
            if imgui.radio_button(f"{label}##upscale", self._mode_idx[0] == i):
                self._mode_idx[0] = i

        imgui.spacing()

        imgui.text("Input Folder")
        imgui.same_line()
        imgui.set_next_item_width(400)
        _, self._folder[0] = imgui.input_text("##upscale_folder", self._folder[0])
        imgui.same_line()
        if imgui.button("Browse##upscale"):
            from src.ui_imgui.widgets.file_dialog import open_folder

            folder = open_folder("Select Folder to Enhance")
            if folder:
                self._folder[0] = folder

        imgui.text("Sample Rate")
        imgui.same_line()
        imgui.set_next_item_width(100)
        _, self._sample_rate_idx[0] = imgui.combo("##sample_rate", self._sample_rate_idx[0], self._sample_rates)

        changed, val = imgui.checkbox("Include Subdirectories##upscale", cfg.get(cfg.include_subdir))
        if changed:
            cfg.set(cfg.include_subdir, val)

        changed, val = imgui.checkbox("Replace Existing##upscale", cfg.get(cfg.replace_existing))
        if changed:
            cfg.set(cfg.replace_existing, val)

        imgui.spacing()
        if imgui.button("Bulk Enhance", size=(150, 0)):
            self._start_upscale()

        self._drawer.draw()

    def _draw_settings_drawer(self):
        from imgui_bundle import imgui

        imgui.text("Bulk Enhancement Settings")
        imgui.separator()
        changed, val = imgui.checkbox("Include Subdirectories", cfg.get(cfg.include_subdir))
        if changed:
            cfg.set(cfg.include_subdir, val)
        changed, val = imgui.checkbox("Replace Existing", cfg.get(cfg.replace_existing))
        if changed:
            cfg.set(cfg.replace_existing, val)
        imgui.text_disabled("The selected mode and sample rate are applied when Bulk Enhance runs.")

    def _draw_help_drawer(self):
        from imgui_bundle import imgui

        imgui.text("Bulk Enhancement Help")
        imgui.separator()
        imgui.text_wrapped("Denoise is intended for recorded speech, Isolate Vocals removes background audio, and Upscale is for low sample-rate input.")
        imgui.text_wrapped("The shared audio player is hidden on this page to keep the batch workflow focused.")

    def _start_upscale(self):
        import os

        folder = self._folder[0]
        if not folder or not os.path.isdir(folder):
            self._state.error_queue.put(("Error", "Please select a valid folder."))
            return

        mode = self._modes[self._mode_idx[0]]
        sample_rate = int(self._sample_rates[self._sample_rate_idx[0]])
        self._state.loading = True
        self._state.loading_message = f"Enhancing audio ({mode})..."

        def _run():
            try:
                from src.utils.model_utils import load_upscaler

                if self._state.upscale_engine is None:
                    load_upscaler(self._state.callbacks)
                if self._state.upscale_engine:
                    self._state.upscale_engine.upscale_dir(
                        self._state.callbacks,
                        folder,
                        mode,
                        sample_rate,
                        cfg.get(cfg.include_subdir),
                        cfg.get(cfg.replace_existing),
                    )
            except Exception as e:
                self._state.error_queue.put(("Upscale Error", str(e)))
            finally:
                self._state.loading = False
                self._state.loading_message = ""

        threading.Thread(target=_run, daemon=True).start()
