from __future__ import annotations

import ctypes
import json
import logging
import os
import queue
import sys

from imgui_bundle import hello_imgui, imgui, immapp, icons_fontawesome_6 as fa

from src.config.config import cfg, VERSION
from src.enums.engine_type import EngineType
from src.ui_imgui.nav_panel import NavPanel
from src.ui_imgui.toolbar import Toolbar
from src.ui_imgui.shared_panels import SharedPanels
from src.ui_imgui.state import AppState, AppCallbacks
from src.ui_imgui.widgets.common import (
    drain_error_queue,
    draw_error_flyouts,
    draw_loader_overlay,
    get_accent_color,
    scale_color,
)
from src.utils.filesystem_utils import get_app_root


def _darken(color: tuple[float, float, float, float], factor: float) -> tuple[float, float, float, float]:
    return (color[0] * factor, color[1] * factor, color[2] * factor, color[3])


def _set_window_icon():
    """Set the GLFW window icon without changing Hello ImGui asset lookup."""
    try:
        from PIL import Image

        icon_candidates = [
            os.path.join(get_app_root(), "resource", "app_settings", "icon.ico"),
            os.path.join(get_app_root(), "resource", "falltalk.ico"),
        ]
        icon_path = next((path for path in icon_candidates if os.path.exists(path)), None)
        if not icon_path:
            return

        window_address = hello_imgui.get_glfw_window_address()
        if not window_address:
            return

        import imgui_bundle

        glfw_dll = os.path.join(os.path.dirname(imgui_bundle.__file__), "glfw3.dll")
        glfw = ctypes.CDLL(glfw_dll)

        img = Image.open(icon_path).convert("RGBA")
        width, height = img.size
        pixels = img.tobytes()

        class GLFWimage(ctypes.Structure):
            _fields_ = [
                ("width", ctypes.c_int),
                ("height", ctypes.c_int),
                ("pixels", ctypes.POINTER(ctypes.c_ubyte)),
            ]

        pixel_array = (ctypes.c_ubyte * len(pixels))(*pixels)
        glfw_image = GLFWimage(width, height, pixel_array)
        window_ptr = ctypes.cast(window_address, ctypes.c_void_p)
        glfw.glfwSetWindowIcon(window_ptr, 1, ctypes.byref(glfw_image))
    except Exception as exc:
        logging.getLogger("falltalk").warning(f"Could not set window icon: {exc}")


def _set_native_dark_title_bar():
    """Request the native Windows title bar/menu chrome to use dark mode."""
    if os.name != "nt":
        return

    try:
        window_address = hello_imgui.get_glfw_window_address()
        if not window_address:
            return

        import imgui_bundle

        dwmapi = ctypes.WinDLL("dwmapi")
        glfw_dll = os.path.join(os.path.dirname(imgui_bundle.__file__), "glfw3.dll")
        glfw = ctypes.CDLL(glfw_dll)
        glfw.glfwGetWin32Window.restype = ctypes.c_void_p
        glfw.glfwGetWin32Window.argtypes = [ctypes.c_void_p]
        hwnd = glfw.glfwGetWin32Window(ctypes.c_void_p(window_address))
        if not hwnd:
            return

        value = ctypes.c_int(1)
        size = ctypes.sizeof(value)

        # Try the newer and older immersive dark mode attributes.
        for attribute in (20, 19):
            try:
                hr = dwmapi.DwmSetWindowAttribute(
                    ctypes.c_void_p(hwnd),
                    ctypes.c_int(attribute),
                    ctypes.byref(value),
                    ctypes.c_int(size),
                )
                if hr == 0:
                    break
            except Exception:
                continue
    except Exception as exc:
        logging.getLogger("falltalk").warning(f"Could not enable dark title bar: {exc}")


def _apply_darcula_theme():
    """Apply the darker ModKit-style New Vegas palette across the shared shell."""
    tweaked = hello_imgui.ImGuiTweakedTheme()
    tweaked.theme = hello_imgui.ImGuiTheme_.darcula
    hello_imgui.apply_tweaked_theme(tweaked)
    style = imgui.get_style()
    accent_tuple = get_accent_color()
    inactive_tuple = (0.4627, 0.4549, 0.3333, 1.0)  # #767455
    highlight_tuple = (0.7216, 0.7020, 0.4784, 1.0)  # #b8b37a
    button_tuple = _darken(accent_tuple, 0.40)
    button_hover_tuple = _darken(accent_tuple, 0.60)
    header_tuple = _darken(accent_tuple, 0.35)
    header_hover_tuple = _darken(accent_tuple, 0.50)
    header_active_tuple = _darken(highlight_tuple, 0.60)
    tab_tuple = _darken(inactive_tuple, 0.40)
    tab_selected_tuple = _darken(accent_tuple, 0.50)
    tab_dimmed_tuple = _darken(inactive_tuple, 0.60)
    text_selected_tuple = (*accent_tuple[:3], 0.40)
    docking_preview_tuple = (*accent_tuple[:3], 0.70)

    accent = imgui.ImVec4(*accent_tuple)
    accent_hover = imgui.ImVec4(*highlight_tuple)
    style.window_rounding = 4.0
    style.child_rounding = 4.0
    style.frame_rounding = 2.0
    style.grab_rounding = 2.0
    style.scrollbar_rounding = 4.0
    style.frame_border_size = 1.0
    style.window_border_size = 1.0
    set_color = style.set_color_
    color_map = {
        imgui.Col_.window_bg: imgui.ImVec4(0.12, 0.12, 0.14, 1.0),
        imgui.Col_.child_bg: imgui.ImVec4(0.10, 0.10, 0.12, 1.0),
        imgui.Col_.popup_bg: imgui.ImVec4(0.14, 0.14, 0.16, 1.0),
        imgui.Col_.border: imgui.ImVec4(0.28, 0.28, 0.30, 1.0),
        imgui.Col_.menu_bar_bg: imgui.ImVec4(0.14, 0.14, 0.16, 1.0),
        imgui.Col_.scrollbar_bg: imgui.ImVec4(0.10, 0.10, 0.12, 1.0),
        imgui.Col_.text: imgui.ImVec4(0.85, 0.85, 0.85, 1.0),
        imgui.Col_.text_disabled: imgui.ImVec4(0.50, 0.50, 0.50, 1.0),
        imgui.Col_.separator: imgui.ImVec4(0.28, 0.28, 0.30, 1.0),
        imgui.Col_.frame_bg: imgui.ImVec4(0.18, 0.18, 0.20, 1.0),
        imgui.Col_.frame_bg_hovered: imgui.ImVec4(0.22, 0.22, 0.25, 1.0),
        imgui.Col_.frame_bg_active: imgui.ImVec4(0.25, 0.25, 0.28, 1.0),
        imgui.Col_.title_bg: imgui.ImVec4(0.10, 0.10, 0.12, 1.0),
        imgui.Col_.title_bg_active: imgui.ImVec4(*tab_tuple),
        imgui.Col_.title_bg_collapsed: imgui.ImVec4(0.10, 0.10, 0.12, 0.75),
        imgui.Col_.button: imgui.ImVec4(*button_tuple),
        imgui.Col_.button_hovered: imgui.ImVec4(*button_hover_tuple),
        imgui.Col_.button_active: imgui.ImVec4(*accent_tuple),
        imgui.Col_.header: imgui.ImVec4(*header_tuple),
        imgui.Col_.header_hovered: imgui.ImVec4(*header_hover_tuple),
        imgui.Col_.header_active: imgui.ImVec4(*header_active_tuple),
        imgui.Col_.tab: imgui.ImVec4(*tab_tuple),
        imgui.Col_.tab_hovered: imgui.ImVec4(*button_hover_tuple),
        imgui.Col_.tab_selected: imgui.ImVec4(*tab_selected_tuple),
        imgui.Col_.tab_selected_overline: accent_hover,
        imgui.Col_.tab_dimmed: imgui.ImVec4(*tab_tuple),
        imgui.Col_.tab_dimmed_selected: imgui.ImVec4(*tab_dimmed_tuple),
        imgui.Col_.tab_dimmed_selected_overline: imgui.ImVec4(*_darken(highlight_tuple, 0.50)),
        imgui.Col_.check_mark: accent,
        imgui.Col_.slider_grab: accent,
        imgui.Col_.slider_grab_active: accent_hover,
        imgui.Col_.scrollbar_grab: imgui.ImVec4(*tab_dimmed_tuple),
        imgui.Col_.scrollbar_grab_hovered: imgui.ImVec4(*inactive_tuple),
        imgui.Col_.scrollbar_grab_active: accent,
        imgui.Col_.resize_grip: imgui.ImVec4(*_darken(inactive_tuple, 0.50)),
        imgui.Col_.resize_grip_hovered: imgui.ImVec4(*inactive_tuple),
        imgui.Col_.resize_grip_active: accent,
        imgui.Col_.separator_hovered: imgui.ImVec4(*inactive_tuple),
        imgui.Col_.separator_active: accent,
        imgui.Col_.text_selected_bg: imgui.ImVec4(*text_selected_tuple),
        imgui.Col_.docking_preview: imgui.ImVec4(*docking_preview_tuple),
        imgui.Col_.docking_empty_bg: imgui.ImVec4(0.08, 0.08, 0.10, 1.0),
        imgui.Col_.drag_drop_target: accent_hover,
        imgui.Col_.table_header_bg: imgui.ImVec4(*tab_tuple),
        imgui.Col_.table_border_strong: imgui.ImVec4(0.28, 0.28, 0.30, 1.0),
        imgui.Col_.table_border_light: imgui.ImVec4(0.22, 0.22, 0.24, 1.0),
        imgui.Col_.table_row_bg: imgui.ImVec4(0.0, 0.0, 0.0, 0.0),
        imgui.Col_.table_row_bg_alt: imgui.ImVec4(1.0, 1.0, 1.0, 0.02),
        imgui.Col_.plot_lines: accent,
        imgui.Col_.plot_lines_hovered: accent_hover,
        imgui.Col_.plot_histogram: accent,
        imgui.Col_.plot_histogram_hovered: accent_hover,
    }
    for col, rgba in color_map.items():
        set_color(col, rgba)


def _build_pages(state: AppState) -> list:
    """Import and instantiate all pages."""
    from src.ui_imgui.pages import (
        CharactersPage, ReferencesPage, GenerationPage, MultiGenerationPage,
        ChatPage, BulkGenerationPage, UpscalePage, EzVoicePage, FaqPage, SettingsPage,
    )
    return [
        CharactersPage(state),
        ReferencesPage(state),
        GenerationPage(state),
        MultiGenerationPage(state),
        ChatPage(state),
        BulkGenerationPage(state),
        UpscalePage(state),
        EzVoicePage(state),
        FaqPage(state),
        SettingsPage(state),
    ]


def run():
    """Entry point for the imgui FallTalk app."""
    # -- Check api_only_mode --
    if cfg.get(cfg.api_only_mode):
        _run_api_only()
        return

    # -- Build shared state --
    state = AppState()
    state.callbacks = AppCallbacks.from_state(state)
    state.engine_type = EngineType(cfg.get(cfg.engine))

    # -- Load static JSON data --
    _load_static_data(state)
    _populate_references(state, "")

    # -- Build pages --
    pages = _build_pages(state)
    shared = SharedPanels(state)
    toolbar = Toolbar(state)

    active_page_id: list[str] = [pages[0].page_id]

    def switch_page(page_id: str):
        if page_id == active_page_id[0]:
            return
        for p in pages:
            if p.page_id == page_id:
                p.on_activate()
            elif p.page_id == active_page_id[0]:
                p.on_deactivate()
        active_page_id[0] = page_id
        nav_panel.set_active(page_id)

    nav_panel = NavPanel(state, pages, switch_page)

    # -- hello_imgui RunnerParams --
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = f"FallTalk - {VERSION}"
    runner_params.app_window_params.window_geometry.size = (1280, 900)
    runner_params.imgui_window_params.show_menu_bar = True
    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    )
    # We don't use docking tabs — hide the tab bar when only one window is docked
    runner_params.docking_params.main_dock_space_node_flags = (
        imgui.DockNodeFlags_.auto_hide_tab_bar
        | imgui.DockNodeFlags_.no_docking_split
    )

    # Load default font with FontAwesome icons
    runner_params.callbacks.load_additional_fonts = (
        hello_imgui.imgui_default_settings.load_default_font_with_font_awesome_icons
    )

    # Top toolbar (edge toolbar, rendered by hello_imgui outside the main content area)
    edge_toolbar_options = hello_imgui.EdgeToolbarOptions()
    edge_toolbar_options.size_em = Toolbar.SIZE_EM
    runner_params.callbacks.add_edge_toolbar(
        hello_imgui.EdgeToolbarType.top,
        lambda: toolbar.draw(),
        edge_toolbar_options,
    )

    _theme_signature = [None]
    api_server = [None]

    def post_init():
        shared.initialize()
        for p in pages:
            p.initialize()
        pages[0].on_activate()
        _set_window_icon()
        _set_native_dark_title_bar()
        _startup_checks(state)
        try:
            from src.api.falltalkapi import FallTalkAPI
            api_server[0] = FallTalkAPI(state)
        except Exception:
            pass

    def show_menus():
        if imgui.begin_menu("View"):
            if imgui.menu_item("Toggle Navigation", "Ctrl+B", False)[0]:
                state.nav_collapsed = not state.nav_collapsed
                cfg.set(cfg.nav_collapsed, state.nav_collapsed)
            if imgui.menu_item("Toggle Audio Panel", "", False)[0]:
                state.audio_panel_expanded = not state.audio_panel_expanded
            imgui.end_menu()

    def main_window_gui():
        """Content of the main dockable window."""
        # Disable parent window scrollbar — children handle their own
        imgui.set_next_window_content_size(imgui.ImVec2(0, 0))

        # Find active page
        active_page = None
        for p in pages:
            if p.page_id == active_page_id[0]:
                active_page = p
                break

        # Update toolbar label for the edge toolbar callback
        toolbar.set_page_label(active_page.label if active_page else "")

        avail = imgui.get_content_region_avail()
        show_audio = active_page.show_audio if active_page else True

        # Nav panel (left)
        nav_panel.draw()
        imgui.same_line(0.0, imgui.get_style().item_spacing.x * 1.5)

        # Content + audio (right) — framed container fills remaining space
        imgui.begin_child("##right_area", size=imgui.ImVec2(0, avail.y))
        try:
            if show_audio:
                audio_h = shared.total_height
                content_h = max(
                    0.0,
                    imgui.get_content_region_avail().y - audio_h - imgui.get_style().item_spacing.y,
                )
                imgui.begin_child(
                    "##content",
                    size=imgui.ImVec2(0, content_h),
                    child_flags=imgui.ChildFlags_.borders,
                )
                try:
                    if active_page:
                        active_page.draw()
                finally:
                    imgui.end_child()

                imgui.spacing()
                shared.draw()
            else:
                imgui.begin_child(
                    "##content",
                    size=imgui.ImVec2(0, 0),
                    child_flags=imgui.ChildFlags_.borders,
                )
                try:
                    if active_page:
                        active_page.draw()
                finally:
                    imgui.end_child()
        finally:
            imgui.end_child()

    def show_gui():
        current_signature = cfg.get(cfg.theme_color)
        if _theme_signature[0] != current_signature:
            _apply_darcula_theme()
            _theme_signature[0] = current_signature

        # Drain queues
        drain_error_queue(state)
        _drain_api_queue(state)

        # Draw overlay widgets. Errors render last so they stay visible even if a loader is active.
        draw_loader_overlay(state)
        draw_error_flyouts()

        # Poll continue_load_pending — engine just finished, resume pending model load
        if state.continue_load_pending:
            state.continue_load_pending = False
            _continue_pending_load(state)

        # Poll model_just_loaded flag
        if state.model_just_loaded:
            state.model_just_loaded = False
            _on_model_loaded(state, pages, switch_page)

    # Set up the main dockable window
    main_window = hello_imgui.DockableWindow()
    main_window.label = "Main"
    main_window.dock_space_name = "MainDockSpace"
    main_window.gui_function = main_window_gui
    main_window.imgui_window_flags = imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse
    runner_params.docking_params.dockable_windows = [main_window]

    runner_params.callbacks.post_init = post_init
    runner_params.callbacks.show_gui = show_gui
    runner_params.callbacks.show_menus = show_menus

    addons = immapp.AddOnsParams()
    addons.with_implot = True
    immapp.run(runner_params, addons)


def _load_static_data(state: AppState):
    import json
    root = get_app_root()
    with open(os.path.join(root, 'config/characters.json'), 'r', encoding='utf-8') as f:
        state.characters_data = {c['name']: c for c in json.load(f)}
    models_path = os.path.join(root, 'config', 'modelsv2.json')
    if os.path.exists(models_path):
        with open(models_path, 'r', encoding='utf-8') as f:
            state.models = {m['name']: m for m in json.load(f)['characters']}
    dr_path = os.path.join(root, 'config', 'default_references.json')
    if os.path.exists(dr_path):
        with open(dr_path, 'r', encoding='utf-8') as f:
            state.default_references = json.load(f)


def _startup_checks(state: AppState):
    """Run startup gates (disclaimer, fallout path, update check)."""
    if not cfg.get(cfg.accepted_disclaimer):
        state._show_disclaimer = True
    if cfg.get(cfg.fallout_4_directory_check):
        if cfg.get(cfg.fallout_4_directory) == "fallout4.exe not found":
            state._show_fallout_warning = True
    if cfg.get(cfg.check_for_updates):
        from src.ui_imgui.widgets.common import AsyncWorker
        from src.utils.huggingface_utils import get_latest_release
        AsyncWorker(get_latest_release).start()


def _populate_references(state: AppState, char_name: str):
    """Populate state.reference_audio from character voicefiles plus custom references."""
    from src.config.config import cfg

    state.reference_audio = []
    state.selected_references = set()
    state.reference_audio_length = 0.0
    if char_name and not char_name.startswith("custom_"):
        import re
        lookup = re.sub(r'_custom\d*$', '', char_name) if '_custom' in char_name else char_name
        char_data = state.characters_data.get(lookup, {})
        for vf in char_data.get('voicefiles', []):
            state.reference_audio.append({
                'filename': vf.get('filename', ''),
                'dialogue': vf.get('dialogue', ''),
                'plugin': vf.get('plugin', ''),
                'folder': lookup,
                'arcname': vf.get('arcname', ''),
            })

    custom_ref_dir = cfg.get(cfg.custom_references) or os.path.join(get_app_root(), "references")
    if os.path.isdir(custom_ref_dir):
        custom_files = [
            os.path.join(custom_ref_dir, name)
            for name in sorted(os.listdir(custom_ref_dir), key=str.lower)
            if os.path.isfile(os.path.join(custom_ref_dir, name))
        ]
        state.reference_audio.extend(custom_files)


def _continue_pending_load(state: AppState):
    """Engine finished loading — if there's a pending character, load it now."""
    import threading
    from src.utils.model_utils import load_model

    char = state.pending_character
    if char is None:
        return

    model = state.pending_model
    rvc = state.pending_rvc
    base = state.pending_base
    display = model.get('display_name', char) if model else char

    # Extract engine version from model data
    engine_version = '1'
    if model and state.engine_type:
        engine_info = model.get(state.engine_type.value, {})
        if isinstance(engine_info, dict):
            engine_version = engine_info.get('engine_version', '1')

    # Clear pending state
    state.pending_character = None
    state.pending_model = None
    state.pending_rvc = None
    state.pending_base = False

    # Populate reference audio table
    _populate_references(state, char)

    state.loading = True
    state.loading_message = f"Loading {display}..."
    threading.Thread(
        target=load_model,
        args=(state.callbacks, char, rvc, display, base, engine_version),
        daemon=True,
    ).start()


def _on_model_loaded(state: AppState, pages: list, switch_page):
    """Handle model_just_loaded flag -- notify all pages."""
    target_page = "generation" if state.engine_type == EngineType.RVC else "references"
    switch_page(target_page)
    for p in pages:
        if hasattr(p, 'on_model_loaded'):
            p.on_model_loaded()


def _drain_api_queue(state: AppState):
    """Drain commands posted by falltalkapi.py."""
    while not state.api_command_queue.empty():
        try:
            cmd = state.api_command_queue.get_nowait()
            _handle_api_command(state, cmd)
        except Exception:
            break


def _handle_api_command(state: AppState, cmd: dict):
    action = cmd.get('action')
    if action == 'engine_change':
        engine = cmd.get('engine')
        if engine:
            cfg.set(cfg.engine, engine)
            state.engine_type = EngineType(engine)


def _run_api_only():
    """Start FastAPI server headlessly (no UI)."""
    from src.api.falltalkapi import FallTalkAPI, app as fastapi_app
    import uvicorn
    FallTalkAPI(None)
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
