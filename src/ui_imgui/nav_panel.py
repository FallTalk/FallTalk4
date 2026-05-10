from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.page import Page
    from src.ui_imgui.state import AppState


def _resolve_icon(icon_str: str) -> str:
    """Convert 'fa-xxx-yyy' to the FontAwesome 6 unicode constant, e.g. ICON_FA_XXX_YYY."""
    if icon_str.startswith("fa-"):
        from imgui_bundle import icons_fontawesome_6 as fa
        attr = "ICON_FA_" + icon_str[3:].upper().replace("-", "_")
        return getattr(fa, attr, fa.ICON_FA_CIRCLE_DOT)
    return icon_str or "•"


class NavPanel:
    """
    Left navigation panel with text labels, icons, grouped items, and separators.
    Replaces the 48px icon-only ActivityBar.
    """
    COLLAPSED_WIDTH = 60
    MIN_WIDTH = 190
    MAX_WIDTH = 240

    def __init__(self, state: AppState, pages: list[Page], on_switch: Callable[[str], None]):
        self._state = state
        self._pages = pages
        self._on_switch = on_switch
        self._active_id: Optional[str] = pages[0].page_id if pages else None

    @property
    def active_id(self) -> Optional[str]:
        return self._active_id

    def set_active(self, page_id: str):
        self._active_id = page_id

    def draw(self):
        from imgui_bundle import imgui

        avail_h = imgui.get_content_region_avail().y
        collapsed = self._state.nav_collapsed
        panel_w = (
            self.COLLAPSED_WIDTH
            if collapsed
            else max(self.MIN_WIDTH, min(self.MAX_WIDTH, int(imgui.get_content_region_avail().x * 0.19)))
        )
        imgui.begin_child("##nav_panel", size=imgui.ImVec2(panel_w, avail_h), child_flags=imgui.ChildFlags_.borders)
        try:
            if not collapsed:
                imgui.text_disabled("Workspace")
                imgui.separator()

            # Collect top and bottom pages
            top_pages = [p for p in self._pages if p.nav_position == "top"]
            bottom_pages = [p for p in self._pages if p.nav_position == "bottom"]

            # Draw top pages grouped with separators
            self._draw_grouped(top_pages, panel_w, collapsed)

            # Push bottom items to bottom
            if bottom_pages:
                # Calculate space needed for bottom items
                item_h = imgui.get_frame_height() + imgui.get_style().item_spacing.y + 4
                bottom_h = len(bottom_pages) * item_h
                # Check groups in bottom pages for additional separators
                groups = sorted(set(p.nav_group for p in bottom_pages))
                if len(groups) > 1:
                    bottom_h += (len(groups) - 1) * (imgui.get_style().item_spacing.y + 2)
                if top_pages and not collapsed:
                    bottom_h += imgui.get_frame_height() + imgui.get_style().item_spacing.y

                current_y = imgui.get_cursor_pos_y()
                target_y = avail_h - bottom_h - imgui.get_style().window_padding.y
                if target_y > current_y:
                    imgui.set_cursor_pos_y(target_y)

                if not collapsed:
                    imgui.separator()
                self._draw_grouped(bottom_pages, panel_w, collapsed)
        finally:
            imgui.end_child()

    def _draw_grouped(self, pages: list[Page], panel_w: float, collapsed: bool):
        from imgui_bundle import imgui

        if not pages:
            return

        prev_group = pages[0].nav_group
        for page in pages:
            if page.nav_group != prev_group and not collapsed:
                imgui.separator()
                prev_group = page.nav_group

            is_active = page.page_id == self._active_id
            w = panel_w - imgui.get_style().window_padding.x * 2
            button_h = imgui.get_frame_height() + 4

            if is_active:
                # Draw left accent bar
                draw_list = imgui.get_window_draw_list()
                cursor = imgui.get_cursor_screen_pos()
                bar_h = button_h
                draw_list.add_rect_filled(
                    imgui.ImVec2(cursor.x, cursor.y),
                    imgui.ImVec2(cursor.x + 3, cursor.y + bar_h),
                    imgui.get_color_u32(imgui.Col_.button_active),
                )
                imgui.push_style_color(imgui.Col_.button, imgui.get_style_color_vec4(imgui.Col_.button_active))

            icon = _resolve_icon(page.icon)
            label = f"{icon}##nav_{page.page_id}" if collapsed else f"  {icon}  {page.label}##nav_{page.page_id}"
            if imgui.button(label, size=imgui.ImVec2(w, button_h)):
                if page.page_id != self._active_id:
                    self._active_id = page.page_id
                    self._on_switch(page.page_id)
            if collapsed and imgui.is_item_hovered():
                imgui.set_tooltip(page.label)

            if is_active:
                imgui.pop_style_color()
