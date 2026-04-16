"""
tkinter-backed file and folder pickers.
Import is lazy so tkinter root is never created unless actually used.
"""
from __future__ import annotations
from typing import Optional

_tk_root = None


def _get_tk():
    global _tk_root
    if _tk_root is None:
        import tkinter as tk
        _tk_root = tk.Tk()
        _tk_root.withdraw()
    return _tk_root


def open_file(title: str = "Open File", filetypes: list[tuple] = None) -> Optional[str]:
    """Return selected file path or None."""
    from tkinter import filedialog
    _get_tk()
    ft = filetypes or [("All Files", "*.*")]
    path = filedialog.askopenfilename(title=title, filetypes=ft)
    return path or None


def open_files(title: str = "Open Files", filetypes: list[tuple] = None) -> list[str]:
    """Return list of selected file paths."""
    from tkinter import filedialog
    _get_tk()
    ft = filetypes or [("All Files", "*.*")]
    paths = filedialog.askopenfilenames(title=title, filetypes=ft)
    return list(paths) if paths else []


def open_folder(title: str = "Select Folder") -> Optional[str]:
    """Return selected folder path or None."""
    from tkinter import filedialog
    _get_tk()
    path = filedialog.askdirectory(title=title)
    return path or None


def save_file(title: str = "Save File", filetypes: list[tuple] = None,
              default_ext: str = "") -> Optional[str]:
    """Return save path or None."""
    from tkinter import filedialog
    _get_tk()
    ft = filetypes or [("All Files", "*.*")]
    path = filedialog.asksaveasfilename(title=title, filetypes=ft,
                                        defaultextension=default_ext)
    return path or None
