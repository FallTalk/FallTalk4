"""Persistent panels that survive page switches."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState


class SharedPanels:
    """Container for panels that stay alive across page switches."""

    def __init__(self, state: AppState):
        self._state = state
        self._audio_panel = None
        self._recorder_panel = None

    def initialize(self):
        from src.ui_imgui.widgets.audio_panel import AudioPanel
        from src.ui_imgui.widgets.recorder_panel import RecorderPanel
        self._audio_panel = AudioPanel(self._state)
        self._recorder_panel = RecorderPanel(self._state)

    @property
    def total_height(self) -> int:
        h = self._audio_panel.height if self._audio_panel else 52
        if self._recorder_panel and self._recorder_panel.should_show:
            h += self._recorder_panel.HEIGHT
        # Account for child border padding + item spacing between content and audio
        h += 8
        return h

    def draw(self):
        from imgui_bundle import imgui
        if self._audio_panel is None:
            return

        imgui.begin_child(
            "##audio_panel", size=imgui.ImVec2(0, 0),
            child_flags=imgui.ChildFlags_.borders,
            window_flags=imgui.WindowFlags_.no_scrollbar,
        )
        try:
            self._audio_panel.draw()

            if self._recorder_panel and self._recorder_panel.should_show:
                imgui.separator()
                self._recorder_panel.draw()
        finally:
            imgui.end_child()
