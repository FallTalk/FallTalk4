from __future__ import annotations
from typing import Callable, Optional


class Drawer:
    """
    Slide-in overlay panel pinned to the right edge of the viewport.
    Pass a callable to open(); it will be called every frame inside the drawer.
    """
    WIDTH = 500
    ANIM_SPEED = 8.0  # units per second (0->1)

    def __init__(self):
        self._open_pct: float = 0.0
        self._opening: bool = False
        self._content_fn: Optional[Callable] = None
        self._title: str = "Panel"
        self._icon: str = ""
        self._width: int = self.WIDTH

    def open(self, content_fn: Callable, title: str = "Panel", icon: str = "", width: Optional[int] = None):
        self._content_fn = content_fn
        self._title = title
        self._icon = icon
        self._width = width or self.WIDTH
        self._opening = True

    def close(self):
        self._opening = False

    @property
    def is_open(self) -> bool:
        return self._open_pct > 0.0

    def draw(self):
        from imgui_bundle import imgui
        from imgui_bundle import hello_imgui

        dt = hello_imgui.frame_rate()
        dt = 1.0 / dt if dt > 0 else 0.016

        if self._opening:
            self._open_pct = min(1.0, self._open_pct + dt * self.ANIM_SPEED)
        else:
            self._open_pct = max(0.0, self._open_pct - dt * self.ANIM_SPEED)
            if self._open_pct == 0.0:
                self._content_fn = None
                return

        vp = imgui.get_main_viewport()
        w = self._width * self._open_pct
        x = vp.pos.x + vp.size.x - w
        y = vp.pos.y
        h = vp.size.y

        imgui.set_next_window_pos(vp.pos)
        imgui.set_next_window_size(vp.size)
        imgui.set_next_window_bg_alpha(0.28 * self._open_pct)
        overlay_flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_saved_settings
        )
        imgui.begin("##drawer_overlay", flags=overlay_flags)
        try:
            pass
        finally:
            imgui.end()

        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(18, 14))
        imgui.push_style_var(imgui.StyleVar_.window_rounding, 10.0)
        imgui.set_next_window_pos((x, y))
        imgui.set_next_window_size((w, h))
        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_saved_settings
        )
        imgui.begin("##drawer", flags=flags)
        try:
            if self._open_pct >= 0.99:
                if self._icon:
                    imgui.text(self._icon)
                    imgui.same_line()
                imgui.text(self._title)
                close_x = max(18.0, imgui.get_window_width() - 48.0)
                imgui.same_line(close_x)
                if imgui.button("X##drawer_close", size=(30, 0)):
                    self.close()
                imgui.separator()
                imgui.begin_child("##drawer_content", size=imgui.ImVec2(0, 0))
                try:
                    if self._content_fn:
                        self._content_fn()
                finally:
                    imgui.end_child()
        finally:
            imgui.end()
        imgui.pop_style_var(2)

        # Click outside to close
        if self._opening and imgui.is_mouse_clicked(0):
            mx, my = imgui.get_mouse_pos()
            if mx < x:
                self.close()
