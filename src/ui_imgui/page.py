from __future__ import annotations


class Page:
    """Base class for all navigation pages."""

    page_id: str = ""
    label: str = ""
    icon: str = "?"
    nav_group: int = 0       # 0-3, separators drawn between groups
    nav_position: str = "top"  # "top" or "bottom"
    show_audio: bool = True    # whether to show the audio panel on this page

    def initialize(self):
        """Called once after GL context is available."""
        pass

    def draw(self):
        """Render full content area for this page."""
        pass

    def draw_menu(self):
        """Optional menu bar items."""
        pass

    def on_activate(self):
        """Called when this page becomes the active page."""
        pass

    def on_deactivate(self):
        """Called when switching away from this page."""
        pass
