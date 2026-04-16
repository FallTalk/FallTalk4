from __future__ import annotations

import threading
import time
from typing import Callable, Optional, TYPE_CHECKING

from src.config.config import cfg

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState


class AsyncWorker:
    """Run a function in a background daemon thread. Poll .done each frame."""
    def __init__(self, target: Callable, args: tuple = ()):
        self.done = False
        self.error: Optional[Exception] = None
        self._thread = threading.Thread(target=self._run, args=args, daemon=True)
        self._target = target

    def _run(self, *args):
        try:
            self._target(*args)
        except Exception as e:
            self.error = e
        finally:
            self.done = True

    def start(self):
        self._thread.start()
        return self


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def hex_to_rgba(color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Convert a '#RRGGBB' color string into normalized RGBA values."""
    value = (color or "").strip().lstrip("#")
    if len(value) != 6:
        value = "f6c062"
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
        _clamp(alpha),
    )


def get_accent_color(alpha: float = 1.0) -> tuple[float, float, float, float]:
    return hex_to_rgba(cfg.get(cfg.theme_color), alpha)


def rgba_to_hex(color: tuple[float, float, float, float] | tuple[float, float, float]) -> str:
    r, g, b = color[:3]
    return "#{:02x}{:02x}{:02x}".format(
        int(_clamp(r) * 255),
        int(_clamp(g) * 255),
        int(_clamp(b) * 255),
    )


def scale_color(color: tuple[float, float, float, float], factor: float, alpha: float | None = None) -> tuple[float, float, float, float]:
    r, g, b, a = color
    return (
        _clamp(r * factor),
        _clamp(g * factor),
        _clamp(b * factor),
        a if alpha is None else _clamp(alpha),
    )


def push_accent_button_style():
    """Apply the legacy yellow CTA style to the next button(s)."""
    from imgui_bundle import imgui

    accent = get_accent_color()
    imgui.push_style_color(imgui.Col_.button, accent)
    imgui.push_style_color(imgui.Col_.button_hovered, scale_color(accent, 1.08))
    imgui.push_style_color(imgui.Col_.button_active, scale_color(accent, 0.92))


def pop_accent_button_style():
    from imgui_bundle import imgui

    imgui.pop_style_color(3)


def draw_icon_button(icon: str, item_id: str, tooltip: str, width: float = 0.0) -> bool:
    from imgui_bundle import imgui

    clicked = imgui.button(f"{icon}##{item_id}", size=(width, 0))
    if imgui.is_item_hovered():
        imgui.set_tooltip(tooltip)
    return clicked


def draw_page_action_strip(
    settings_id: str,
    help_id: str,
    settings_tooltip: str,
    help_tooltip: str,
    on_settings: Callable[[], None],
    on_help: Callable[[], None],
):
    from imgui_bundle import imgui, icons_fontawesome_6 as fa

    if draw_icon_button(fa.ICON_FA_GEAR, settings_id, settings_tooltip):
        on_settings()
    imgui.same_line()
    if draw_icon_button(fa.ICON_FA_CIRCLE_QUESTION, help_id, help_tooltip):
        on_help()
    imgui.separator()


def draw_loader_overlay(state: AppState):
    """
    Render a full-viewport semi-transparent blocking overlay when state.loading is True.
    Call this at the end of every workspace's draw() method.
    """
    if not state.loading:
        return
    try:
        from imgui_bundle import imgui
        if state.loading_started_at == 0.0:
            state.loading_started_at = time.time()
        if state.loading_message and not state.loading_history:
            state.loading_history.append(state.loading_message)
        elapsed = max(0.0, time.time() - state.loading_started_at) if state.loading_started_at else 0.0
        known_progress = state.loading_progress is not None
        progress = state.loading_progress if known_progress else ((time.time() * 0.6) % 1.0)
        vp = imgui.get_main_viewport()
        imgui.set_next_window_pos(vp.pos)
        imgui.set_next_window_size(vp.size)
        imgui.set_next_window_bg_alpha(0.6)
        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_inputs
        )
        imgui.begin("##loader_overlay", flags=flags)
        try:
            win_w, win_h = imgui.get_window_size()
            panel_w = min(560.0, win_w - 80.0)
            panel_h = 190.0 if state.loading_history else 150.0
            imgui.set_cursor_pos(((win_w - panel_w) * 0.5, (win_h - panel_h) * 0.5))
            imgui.begin_child(
                "##loader_panel",
                size=imgui.ImVec2(panel_w, panel_h),
                child_flags=imgui.ChildFlags_.borders,
            )
            try:
                imgui.text("Working")
                imgui.same_line()
                imgui.text_disabled(f"{int(elapsed // 60):02}:{int(elapsed % 60):02}")
                imgui.separator()

                msg = state.loading_message or "Loading..."
                imgui.text_wrapped(msg)
                imgui.spacing()
                imgui.progress_bar(progress, size=imgui.ImVec2(-1, 16))
                if known_progress:
                    imgui.text_disabled(f"{progress * 100:.0f}% complete")
                else:
                    imgui.text_disabled("Progress will update as stages report back.")

                if state.loading_history:
                    imgui.spacing()
                    imgui.text_disabled("Recent Stages")
                    for entry in state.loading_history[-3:]:
                        imgui.bullet_text(entry)
            finally:
                imgui.end_child()
        finally:
            imgui.end()
    except Exception:
        pass  # Never crash the main loop


_error_display: list[tuple[float, str, str]] = []  # (expire_time, title, msg)


def drain_error_queue(state: AppState):
    """
    Drain state.error_queue and state.warn_queue into the display list.
    Call once per frame before draw_error_flyouts().
    """
    while not state.error_queue.empty():
        try:
            title, msg = state.error_queue.get_nowait()
            _error_display.append((time.time() + 5.0, title, msg))
        except Exception:
            break
    while not state.warn_queue.empty():
        try:
            title, msg = state.warn_queue.get_nowait()
            _error_display.append((time.time() + 5.0, title, msg))
        except Exception:
            break


def draw_error_flyouts():
    """
    Render queued error/warn messages as timed imgui popups.
    Call once per frame after drain_error_queue().
    """
    from imgui_bundle import imgui
    now = time.time()
    # Remove expired
    _error_display[:] = [(exp, t, m) for exp, t, m in _error_display if exp > now]
    for i, (_, title, msg) in enumerate(_error_display):
        vp = imgui.get_main_viewport()
        imgui.set_next_window_pos(
            (vp.pos.x + vp.size.x * 0.5, vp.pos.y + 40),
            imgui.Cond_.always,
            (0.5, 0.0)
        )
        imgui.set_next_window_bg_alpha(0.85)
        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.always_auto_resize
        )
        imgui.begin(f"##err_{i}", flags=flags)
        imgui.text_colored((1.0, 0.4, 0.4, 1.0), f"  {title}  ")
        imgui.text_wrapped(msg)
        imgui.end()
