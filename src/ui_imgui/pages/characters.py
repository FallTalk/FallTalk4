from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState

from src.ui_imgui.page import Page
from src.ui_imgui.widgets.common import draw_icon_button


class CharactersPage(Page):
    page_id = "characters"
    label = "Character Models"
    icon = "fa-users"
    nav_group = 0
    nav_position = "top"
    show_audio = False

    def __init__(self, state: AppState):
        self._state = state
        self._char_filter = [""]
        self._confirm_delete: dict | None = None
        self._rvc_only = [False]
        self._downloaded_only = [False]

    def draw(self):
        from imgui_bundle import imgui
        from src.ui_imgui.widgets.tables import search_filter

        trained = self._get_trained_characters()
        untrained = self._get_untrained_characters()
        custom = self._get_custom_characters()

        imgui.text_disabled("Load, download, and manage models for the active engine.")
        search_filter("char_search", self._char_filter)
        imgui.same_line()
        if imgui.button("Import Custom Model Folder"):
            self._show_import_dialog()
        imgui.same_line()
        changed, self._rvc_only[0] = imgui.checkbox("RVC Only", self._rvc_only[0])
        imgui.same_line()
        changed, self._downloaded_only[0] = imgui.checkbox("Downloaded Only", self._downloaded_only[0])

        imgui.separator()

        if imgui.begin_tab_bar("char_tabs"):
            if imgui.begin_tab_item(f"Trained ({len(trained)})")[0]:
                self._draw_character_table("trained", trained)
                imgui.end_tab_item()
            if imgui.begin_tab_item(f"Untrained ({len(untrained)})")[0]:
                self._draw_character_table("untrained", untrained)
                imgui.end_tab_item()
            if imgui.begin_tab_item(f"Custom ({len(custom)})")[0]:
                self._draw_character_table("custom", custom)
                imgui.end_tab_item()
            imgui.end_tab_bar()

        self._draw_delete_confirmation()

    def _draw_character_table(self, table_id: str, characters: list):
        from imgui_bundle import imgui, icons_fontawesome_6 as fa
        from src.ui_imgui.widgets.tables import begin_table, end_table
        cols = ["Name", "Load", "Download", "Update", "Delete", "RVC"]
        if begin_table(f"char_table_{table_id}", cols):
            flt = self._char_filter[0].lower()
            for char in characters:
                name = char.get('display_name', char.get('name', ''))
                if flt and flt not in name.lower():
                    continue
                if self._rvc_only[0] and char.get('RVC') is None:
                    continue
                if self._downloaded_only[0] and not self._is_downloaded(char):
                    continue
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                imgui.align_text_to_frame_padding()
                imgui.text(name)
                imgui.table_set_column_index(1)
                if draw_icon_button(fa.ICON_FA_PLAY, f"char_load_{name}", "Load"):
                    self._load_character(char)
                imgui.table_set_column_index(2)
                if draw_icon_button(fa.ICON_FA_DOWNLOAD, f"char_download_{name}", "Download"):
                    self._download_character(char)
                imgui.table_set_column_index(3)
                if draw_icon_button(fa.ICON_FA_ARROWS_ROTATE, f"char_update_{name}", "Update"):
                    self._update_character(char)
                imgui.table_set_column_index(4)
                if draw_icon_button(fa.ICON_FA_TRASH, f"char_delete_{name}", "Delete"):
                    self._confirm_delete = char
                    imgui.open_popup("Delete Character?")
                imgui.table_set_column_index(5)
                has_rvc = char.get('RVC') is not None
                if has_rvc:
                    imgui.text_colored((0.4, 1.0, 0.4, 1.0), fa.ICON_FA_CHECK)
                else:
                    imgui.text_colored((0.5, 0.5, 0.5, 1.0), fa.ICON_FA_MINUS)
            end_table()

    def _draw_delete_confirmation(self):
        from imgui_bundle import imgui
        if imgui.begin_popup_modal("Delete Character?", flags=imgui.WindowFlags_.always_auto_resize)[0]:
            if self._confirm_delete:
                name = self._confirm_delete.get('display_name', self._confirm_delete.get('name', ''))
                imgui.text(f"Delete character '{name}'?")
                imgui.text("This cannot be undone.")
                imgui.separator()
                if imgui.button("Yes, Delete"):
                    self._delete_character(self._confirm_delete)
                    self._confirm_delete = None
                    imgui.close_current_popup()
                imgui.same_line()
                if imgui.button("Cancel"):
                    self._confirm_delete = None
                    imgui.close_current_popup()
            imgui.end_popup()

    def _get_trained_characters(self) -> list:
        from src.config.config import cfg
        result = []
        for name, char in self._state.characters_data.items():
            if name in self._state.models and cfg.get(cfg.engine) in self._state.models[name]:
                c = dict(char)
                model_data = self._state.models[name]
                c['display_name'] = model_data.get('display_name', name)
                c['RVC'] = model_data.get('RVC')
                result.append(c)
        return result

    def _get_untrained_characters(self) -> list:
        from src.config.config import cfg
        result = []
        for name, char in self._state.characters_data.items():
            if name not in self._state.models or cfg.get(cfg.engine) not in self._state.models.get(name, {}):
                c = dict(char)
                if 'display_name' not in c:
                    c['display_name'] = name
                c['RVC'] = self._state.models.get(name, {}).get('RVC')
                result.append(c)
        return result

    def _get_custom_characters(self) -> list:
        if not self._state.custom_models:
            return []
        result = []
        for model in self._state.custom_models.values():
            c = dict(model)
            c['RVC'] = model.get('RVC')
            result.append(c)
        return result

    @staticmethod
    def _is_downloaded(char: dict) -> bool:
        from src.utils.filesystem_utils import get_app_root

        model_name = char.get('name', '')
        if char.get('path'):
            return os.path.exists(char['path'])
        return os.path.isdir(os.path.join(get_app_root(), 'models', model_name))

    def _load_character(self, char: dict):
        from src.utils.model_utils import load_model, get_engine_loader
        from src.config.config import cfg
        from src.enums.engine_type import EngineType

        name = char['name']
        display = char.get('display_name', name)
        model_data = self._state.models.get(name, {})
        rvc = model_data.get('RVC')
        cb = self._state.callbacks

        engine_type = EngineType(cfg.get(cfg.engine))
        engine_info = model_data.get(engine_type.value)
        base_model = not bool(engine_info)
        engine_version = engine_info.get('engine_version', '1') if isinstance(engine_info, dict) else '1'

        if self._state.tts_engine is None and cfg.get(cfg.engine) is not None:
            # Engine not loaded yet — store pending info and load engine first
            self._state.pending_character = name
            self._state.pending_model = model_data
            self._state.pending_rvc = rvc
            self._state.pending_base = base_model
            loader = get_engine_loader(engine_type)
            if loader:
                self._state.loading = True
                self._state.loading_message = f"Loading {engine_type.value} engine..."
                threading.Thread(target=loader, args=(cb,), daemon=True).start()
            else:
                self._state.error_queue.put(("Unable to Load", f"No loader for engine {engine_type.value}"))
        else:
            from src.ui_imgui.app import _populate_references
            _populate_references(self._state, name)
            self._state.loading = True
            self._state.loading_message = f"Loading {'base model' if base_model else display}..."
            threading.Thread(
                target=load_model,
                args=(cb, name, rvc, display, base_model, engine_version),
                daemon=True
            ).start()

    def _download_character(self, char: dict):
        from src.utils.huggingface_utils import download_models
        from src.config.config import cfg
        from src.enums.engine_type import EngineType

        model = self._state.models.get(char['name'], {})
        engine_type = EngineType(cfg.get(cfg.engine))
        model = model.get(engine_type.value) if isinstance(model, dict) else None
        if not isinstance(model, dict):
            self._state.error_queue.put(("Unable to Download", f"No {engine_type.value} model metadata found for {char['name']}"))
            return

        rvc = model.get('RVC')
        self._state.loading = True
        self._state.loading_message = f"Downloading {char.get('display_name', char['name'])}..."
        threading.Thread(
            target=download_models,
            args=(self._state.callbacks, char['name'], model, rvc),
            daemon=True
        ).start()

    def _update_character(self, char: dict):
        self._download_character(char)

    def _delete_character(self, char: dict):
        import shutil
        from src.utils.filesystem_utils import get_app_root
        name = char.get('name', '')
        model_dir = os.path.join(get_app_root(), 'models', name)
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)

    def _show_import_dialog(self):
        from src.ui_imgui.widgets.file_dialog import open_folder
        path = open_folder("Select Custom Model Folder")
        if path:
            name = os.path.basename(path)
            if self._state.custom_models is None:
                self._state.custom_models = {}
            self._state.custom_models[name] = {
                'name': name,
                'display_name': name,
                'path': path,
            }
