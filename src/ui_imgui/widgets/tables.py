"""
Reusable imgui table helpers for character lists, reference tables, bulk CSV tables.
"""
from __future__ import annotations
from typing import Any


def begin_table(table_id: str, columns: list[str], flags=None) -> bool:
    """
    Begin an imgui table with the given column headers.
    Returns True if the table was begun (caller must call end_table()).
    """
    from imgui_bundle import imgui
    if flags is None:
        flags = (
            imgui.TableFlags_.resizable
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.scroll_y
            | imgui.TableFlags_.sort_multi
        )
    imgui.push_style_var(imgui.StyleVar_.cell_padding, imgui.ImVec2(10, 10))
    if imgui.begin_table(table_id, len(columns), flags):
        for col in columns:
            imgui.table_setup_column(col)
        imgui.table_headers_row()
        return True
    imgui.pop_style_var()
    return False


def end_table():
    from imgui_bundle import imgui
    imgui.end_table()
    imgui.pop_style_var()


def search_filter(filter_id: str, filter_text: list[str], width: float = 200.0) -> bool:
    """
    Render a search input. filter_text is a 1-element list (mutable string).
    Returns True if the value changed.
    """
    from imgui_bundle import imgui
    imgui.set_next_item_width(width)
    changed, new_val = imgui.input_text(f"Search##{filter_id}", filter_text[0])
    if changed:
        filter_text[0] = new_val
    return changed
