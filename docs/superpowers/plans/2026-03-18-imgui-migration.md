# imgui_bundle Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all PySide6/qfluentwidgets UI code with imgui_bundle, achieving full feature parity while keeping the `cfg.get()/cfg.set()` API and all non-UI logic unchanged.

**Architecture:** Big-bang rewrite into `src/ui_imgui/` using the hello_imgui workspace protocol (modeled on `I:\Fallout4Mods\Fallout4MCP\ui`). A single `AppState` object replaces the Qt app's scattered `self.*` state. `AppCallbacks` replaces all `QMetaObject.invokeMethod` cross-thread calls in `src/utils/`.

**Tech Stack:** `imgui-bundle>=1.5.0` (includes hello_imgui + implot), `sounddevice`, `soundfile`, `tkinter` (stdlib, for file dialogs), `configparser` (stdlib, for migrating old QSettings config), `pytest`

---

## Phase Execution Order

```
Phase 1 (Foundation) → must complete first
Phase 2 (Streams 2–6) → run in parallel after Phase 1
Phase 3 (Integration) → run after all Phase 2 streams complete
```

---

## File Map

### Created (new)
| File | Responsibility |
|---|---|
| `src/ui_imgui/__init__.py` | Package marker |
| `src/ui_imgui/app.py` | hello_imgui bootstrap, RunnerParams, Darcula theme, startup flow |
| `src/ui_imgui/workspace.py` | Workspace protocol definition |
| `src/ui_imgui/activity_bar.py` | Left icon sidebar for workspace switching |
| `src/ui_imgui/shared_panels.py` | AudioPanel registration |
| `src/ui_imgui/state.py` | AppState dataclass + AppCallbacks dataclass |
| `src/ui_imgui/workspaces/__init__.py` | Package marker |
| `src/ui_imgui/workspaces/characters.py` | Characters + References panels |
| `src/ui_imgui/workspaces/generation.py` | Generation + Multi-gen + RVC panels |
| `src/ui_imgui/workspaces/bulk.py` | Bulk CSV + RVC + FUZ + EzVoice panels |
| `src/ui_imgui/workspaces/chat.py` | Chat panel |
| `src/ui_imgui/workspaces/upscale.py` | Upscale panel |
| `src/ui_imgui/workspaces/settings.py` | Global settings + FAQ panels |
| `src/ui_imgui/widgets/__init__.py` | Package marker |
| `src/ui_imgui/widgets/engine_settings.py` | Replaces all 20 `src/settings/*.py` files |
| `src/ui_imgui/widgets/engine_help.py` | Replaces all 20+ `src/help/*.py` files |
| `src/ui_imgui/widgets/drawer.py` | Slide-in overlay panel |
| `src/ui_imgui/widgets/audio_panel.py` | Dockable waveform player + recorder |
| `src/ui_imgui/widgets/tables.py` | Reusable imgui table helpers |
| `src/ui_imgui/widgets/file_dialog.py` | tkinter-backed file/folder pickers |
| `src/ui_imgui/widgets/common.py` | Loader overlay, error flyouts, AsyncWorker |
| `tests/__init__.py` | Package marker |
| `tests/test_config.py` | Unit tests for config migration |
| `tests/test_state.py` | Unit tests for AppState/AppCallbacks |

### Modified (existing)
| File | Change |
|---|---|
| `src/config/config.py` | Full internal rewrite — strip Qt, keep `cfg.get()/cfg.set()` API |
| `src/utils/model_utils.py` | Replace `QMetaObject.invokeMethod` with `parent.<callback>()` |
| `src/utils/inference_utils.py` | Same |
| `src/utils/bulk_utils.py` | Same |
| `src/utils/huggingface_utils.py` | Same |
| `src/api/falltalkapi.py` | Replace `QMetaObject.invokeMethod` with `state.api_command_queue` |
| `FallTalk.py` | Switch entry point to `src/ui_imgui/app.py` |

### Deleted (after Phase 3)
`src/widgets/` (21 files), `src/settings/` (20 files), `src/help/` (20+ files), `src/ui/cards.py`

---

## Phase 1: Foundation

> All parallel streams depend on this phase. Complete all tasks before starting Phase 2.

---

### Task 1: Test infrastructure + pytest setup

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_config.py` (scaffold only, will be populated in Task 2)
- Create: `pytest.ini`

- [ ] **Step 1: Create pytest.ini**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

- [ ] **Step 2: Create tests/__init__.py**

```python
```

- [ ] **Step 3: Create tests/test_config.py scaffold**

```python
"""Tests for src/config/config.py — populated after Task 2."""
```

- [ ] **Step 4: Run pytest to confirm test collection works**

```bash
pytest --collect-only
```
Expected: `no tests ran` (0 errors, just no tests yet)

- [ ] **Step 5: Commit**

```bash
git add pytest.ini tests/
git commit -m "test: add pytest infrastructure"
```

---

### Task 2: Config system migration

**Spec:** `docs/superpowers/specs/2026-03-18-imgui-migration-design.md` §6

**Files:**
- Modify: `src/config/config.py` (full internal rewrite)
- Modify: `tests/test_config.py`

The current `config.py` uses `qfluentwidgets.QConfig`. Strip all Qt/qfluentwidgets imports and replace internals with a JSON-backed `Config` class while keeping every `cfg.get(cfg.X)` / `cfg.set(cfg.X, value)` call unchanged.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config.py
import os, json, tempfile, pytest
from unittest.mock import patch

def test_config_key_has_key_and_default():
    from src.config.config import ConfigKey
    k = ConfigKey("my_key", "default_val")
    assert k.key == "my_key"
    assert k.default == "default_val"

def test_config_get_returns_default_when_missing():
    from src.config.config import Config, ConfigKey
    with tempfile.TemporaryDirectory() as d:
        with patch("src.config.config.get_app_root", return_value=d):
            os.makedirs(os.path.join(d, "config"), exist_ok=True)
            c = Config()
            k = ConfigKey("missing_key", "hello")
            assert c.get(k) == "hello"

def test_config_set_and_get():
    from src.config.config import Config, ConfigKey
    with tempfile.TemporaryDirectory() as d:
        with patch("src.config.config.get_app_root", return_value=d):
            os.makedirs(os.path.join(d, "config"), exist_ok=True)
            c = Config()
            k = ConfigKey("test_key", 0, int)
            c.set(k, 42)
            assert c.get(k) == 42

def test_config_persists_across_instances():
    from src.config.config import Config, ConfigKey
    with tempfile.TemporaryDirectory() as d:
        with patch("src.config.config.get_app_root", return_value=d):
            os.makedirs(os.path.join(d, "config"), exist_ok=True)
            k = ConfigKey("persist_key", "orig", str)
            c1 = Config()
            c1.set(k, "saved")
            c2 = Config()
            assert c2.get(k) == "saved"

def test_config_reset_restores_defaults():
    from src.config.config import Config, ConfigKey
    with tempfile.TemporaryDirectory() as d:
        with patch("src.config.config.get_app_root", return_value=d):
            os.makedirs(os.path.join(d, "config"), exist_ok=True)
            c = Config()
            k = ConfigKey("reset_key", "orig_default", str)
            c.set(k, "changed")
            c.reset()
            assert c.get(k) == "orig_default"

def test_config_type_coercion():
    from src.config.config import Config, ConfigKey
    with tempfile.TemporaryDirectory() as d:
        with patch("src.config.config.get_app_root", return_value=d):
            os.makedirs(os.path.join(d, "config"), exist_ok=True)
            # Simulate JSON storing a float as string
            settings_path = os.path.join(d, "config", "settings.json")
            with open(settings_path, "w") as f:
                json.dump({"coerce_key": "3.14"}, f)
            c = Config()
            k = ConfigKey("coerce_key", 1.0, float)
            assert c.get(k) == 3.14

def test_config_ini_migration(tmp_path):
    """Old QSettings INI format is migrated on first load."""
    from src.config.config import Config, ConfigKey
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    # Write fake QSettings INI file
    ini = config_dir / "configv2.json"
    ini.write_text("[General]\ndevice=cpu\nspeed=1.5\n")
    with patch("src.config.config.get_app_root", return_value=str(tmp_path)):
        c = Config()
        k_device = ConfigKey("device", "cuda", str)
        assert c.get(k_device) == "cpu"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_config.py -v
```
Expected: All FAIL (ImportError or AttributeError)

- [ ] **Step 3: Rewrite src/config/config.py internals**

Replace the file content. Keep all `ConfigKey` attribute names identical to the current `cfg.X` names. The full rewrite:

```python
# coding: utf-8
import configparser
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.utils.filesystem_utils import get_app_root


def find_fallout4_exe():
    drives = [d + ':\\' for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    common_paths = [
        '\\Program Files (x86)\\Steam\\steamapps\\common\\Fallout 4\\',
        '\\Program Files\\Steam\\steamapps\\common\\Fallout 4\\',
        '\\Program Files (x86)\\Fallout 4\\',
        '\\Program Files\\Fallout 4\\',
        '\\Steam\\steamapps\\common\\Fallout 4\\',
        '\\Steam Games\\steamapps\\common\\Fallout 4\\',
    ]
    for drive in drives:
        if not os.path.exists(drive):
            continue
        for common_path in common_paths:
            exe = os.path.join(drive, common_path.lstrip('\\'), 'Fallout4.exe')
            if os.path.isfile(exe):
                return os.path.dirname(exe)
    return "fallout4.exe not found"


class ConfigKey:
    def __init__(self, key: str, default: Any, type_=None):
        self.key = key
        self.default = default
        self.type_ = type_


class Config:
    # --- General ---
    engine = ConfigKey("engine", "XTTSv2", str)
    load_engine_art_start = ConfigKey("load_engine_art_start", False, bool)
    auto_update_models = ConfigKey("auto_update_models", True, bool)
    device = ConfigKey("device", "cuda", str)
    seed = ConfigKey("seed", -1, int)
    check_for_updates = ConfigKey("check_for_updates", True, bool)
    download_configs = ConfigKey("download_configs", True, bool)
    auto_play = ConfigKey("auto_play", True, bool)
    disableSSLVerify = ConfigKey("disableSSLVerify", False, bool)
    api_only_mode = ConfigKey("api_only_mode", False, bool)
    accepted_disclaimer = ConfigKey("accepted_disclaimer", False, bool)
    accepts_custom_disclaimer = ConfigKey("accepts_custom_disclaimer", False, bool)
    first_start = ConfigKey("first_start", True, bool)

    # --- Paths ---
    fallout_4_directory = ConfigKey("fallout_4_directory", "fallout4.exe not found", str)
    fallout_4_directory_check = ConfigKey("fallout_4_directory_check", True, bool)
    custom_references = ConfigKey("custom_references", None)
    output_dir = ConfigKey("output_dir", "output", str)
    huggingface_cache_dir = ConfigKey("huggingface_cache_dir", None)

    # --- Audio ---
    audio_output_device = ConfigKey("audio_output_device", -1, int)
    audio_input_device = ConfigKey("audio_input_device", -1, int)

    # --- Text processing ---
    max_text_size = ConfigKey("max_text_size", 200, int)
    min_chunk_size = ConfigKey("min_chunk_size", 50, int)
    lowercase_conversion = ConfigKey("lowercase_conversion", False, bool)
    whitespace_normalization = ConfigKey("whitespace_normalization", True, bool)
    dot_letter_fix = ConfigKey("dot_letter_fix", True, bool)
    inline_reference_removal = ConfigKey("inline_reference_removal", True, bool)
    pad_short_phrases = ConfigKey("pad_short_phrases", True, bool)

    # --- Bulk ---
    replace_existing = ConfigKey("replace_existing", False, bool)
    include_subdir = ConfigKey("include_subdir", False, bool)
    threads = ConfigKey("threads", 1, int)
    multigen_total = ConfigKey("multigen_total", 3, int)
    ez_total = ConfigKey("ez_total", 1, int)

    # --- Features ---
    rvc_enabled = ConfigKey("rvc_enabled", False, bool)
    apbwe_enabled = ConfigKey("apbwe_enabled", False, bool)
    xwm_enabled = ConfigKey("xwm_enabled", False, bool)
    keep_only_fuz = ConfigKey("keep_only_fuz", False, bool)
    use_existing_lip = ConfigKey("use_existing_lip", True, bool)

    # --- UI (imgui) ---
    font_global_scale = ConfigKey("font_global_scale", 1.0, float)

    # --- RVC ---
    rvc_pitch = ConfigKey("rvc_pitch", 0, int)
    rvc_hop_length = ConfigKey("rvc_hop_length", 128, int)
    rvc_training_data_size = ConfigKey("rvc_training_data_size", 0, int)
    rvc_index_influence = ConfigKey("rvc_index_influence", 75, int)
    rvc_volume_envelope = ConfigKey("rvc_volume_envelope", 100, int)
    rvc_protect = ConfigKey("rvc_protect", 33, int)
    rvc_filter_radius = ConfigKey("rvc_filter_radius", 3, int)
    rvc_autotune = ConfigKey("rvc_autotune", False, bool)
    rvc_split_audio = ConfigKey("rvc_split_audio", False, bool)
    rvc_pitch_extraction = ConfigKey("rvc_pitch_extraction", "rmvpe", str)
    rvc_mode = ConfigKey("rvc_mode", "Microphone", str)
    rvc_eleven_labs_key = ConfigKey("rvc_eleven_labs_key", "", str)
    rvc_embedder_model = ConfigKey("rvc_embedder_model", "contentvec", str)

    # --- XTTSv2 ---
    speed = ConfigKey("speed", 1, int)
    model_temperature = ConfigKey("model_temperature", 20, int)
    model_repetition = ConfigKey("model_repetition", 10, int)
    low_vram = ConfigKey("low_vram", False, bool)
    deepspeed_enabled = ConfigKey("deepspeed_enabled", False, bool)

    # --- F5 ---
    f5_mode = ConfigKey("f5_mode", "generate", str)
    f5_speed = ConfigKey("f5_speed", 1.0, float)
    f5_nfe_step = ConfigKey("f5_nfe_step", 32, int)
    f5_crossfade = ConfigKey("f5_crossfade", 0.15, float)

    # --- GPT-SoVITS ---
    gpt_sovits_slice_mode = ConfigKey("gpt_sovits_slice_mode", "No slice", str)
    gpt_sovits_low_vram = ConfigKey("gpt_sovits_low_vram", False, bool)
    gpt_sovits_top_p = ConfigKey("gpt_sovits_top_p", 100, int)
    gpt_sovits_top_k = ConfigKey("gpt_sovits_top_k", 5, int)
    gpt_sovits_temperature = ConfigKey("gpt_sovits_temperature", 100, int)
    gpt_sovits_speed = ConfigKey("gpt_sovits_speed", 100, int)

    # --- StyleTTS2 ---
    styletts2_alpha = ConfigKey("styletts2_alpha", 30, int)
    styletts2_beta = ConfigKey("styletts2_beta", 70, int)
    styletts2_embedding_scale = ConfigKey("styletts2_embedding_scale", 100, int)
    styletts2_diffusion_steps = ConfigKey("styletts2_diffusion_steps", 5, int)

    # --- FishSpeech ---
    fish_use_torch_compile = ConfigKey("fish_use_torch_compile", False, bool)
    fish_temperature = ConfigKey("fish_temperature", 70, int)
    fish_repetition = ConfigKey("fish_repetition", 12, int)
    fish_top_p = ConfigKey("fish_top_p", 70, int)
    fish_max_length = ConfigKey("fish_max_length", 2048, int)
    fish_use_cache = ConfigKey("fish_use_cache", False, bool)
    fish_iterative_prompt = ConfigKey("fish_iterative_prompt", True, bool)

    # --- DIA ---
    dia_use_torch_compile = ConfigKey("dia_use_torch_compile", False, bool)
    dia_temperature = ConfigKey("dia_temperature", 135, int)
    dia_top_k = ConfigKey("dia_top_k", 45, int)
    dia_top_p = ConfigKey("dia_top_p", 95, int)

    # --- Llasa ---
    llasa_temperature = ConfigKey("llasa_temperature", 80, int)
    llasa_top_p = ConfigKey("llasa_top_p", 90, int)
    llasa_max_length = ConfigKey("llasa_max_length", 2048, int)
    llasa_mode = ConfigKey("llasa_mode", "1b", str)

    # --- Orpheus ---
    orpheus_temperature = ConfigKey("orpheus_temperature", 85, int)
    orpheus_top_p = ConfigKey("orpheus_top_p", 95, int)
    orpheus_repetition = ConfigKey("orpheus_repetition", 115, int)
    orpheus_top_k = ConfigKey("orpheus_top_k", 50, int)
    orpheus_max_new_tokens = ConfigKey("orpheus_max_new_tokens", 1200, int)

    # --- Spark ---
    spark_top_p = ConfigKey("spark_top_p", 90, int)
    spark_temperature = ConfigKey("spark_temperature", 90, int)
    spark_top_k = ConfigKey("spark_top_k", 50, int)
    spark_max_new_tokens = ConfigKey("spark_max_new_tokens", 3000, int)

    # --- Chatterbox ---
    chatterbox_top_p = ConfigKey("chatterbox_top_p", 80, int)
    chatterbox_temperature = ConfigKey("chatterbox_temperature", 80, int)
    chatterbox_min_p = ConfigKey("chatterbox_min_p", 5, int)
    chatterbox_max_new_tokens = ConfigKey("chatterbox_max_new_tokens", 4096, int)
    chatterbox_exaggeration = ConfigKey("chatterbox_exaggeration", 50, int)
    chatterbox_repetition_penalty = ConfigKey("chatterbox_repetition_penalty", 100, int)
    chatterbox_cfg_weight = ConfigKey("chatterbox_cfg_weight", 50, int)

    # --- Higgs ---
    higgs_top_p = ConfigKey("higgs_top_p", 90, int)
    higgs_temperature = ConfigKey("higgs_temperature", 90, int)
    higgs_top_k = ConfigKey("higgs_top_k", 50, int)
    higgs_max_new_tokens = ConfigKey("higgs_max_new_tokens", 3000, int)
    higgs_ras_win_len = ConfigKey("higgs_ras_win_len", 15, int)
    higgs_ras_win_max_num_repeat = ConfigKey("higgs_ras_win_max_num_repeat", 4, int)

    # --- Vibe ---
    vibe_mode = ConfigKey("vibe_mode", "generate", str)
    vibe_cfg_scale = ConfigKey("vibe_cfg_scale", 10, int)
    vibe_inference_steps = ConfigKey("vibe_inference_steps", 50, int)
    vibe_temperature = ConfigKey("vibe_temperature", 100, int)
    vibe_do_sample = ConfigKey("vibe_do_sample", True, bool)
    vibe_top_p = ConfigKey("vibe_top_p", 90, int)
    vibe_top_k = ConfigKey("vibe_top_k", 50, int)

    # --- Qwen3TTS ---
    qwen_instruct = ConfigKey("qwen_instruct", "", str)
    qwen_language = ConfigKey("qwen_language", "English", str)
    qwen_model_version = ConfigKey("qwen_model_version", "1.7B-Base", str)

    # --- CSM ---
    csm_temperature = ConfigKey("csm_temperature", 80, int)
    csm_top_k = ConfigKey("csm_top_k", 50, int)

    # --- DMO Speech 2 ---
    dmo_temperature = ConfigKey("dmo_temperature", 80, int)
    dmo_top_k = ConfigKey("dmo_top_k", 50, int)

    # --- Chat ---
    chat_model = ConfigKey("chat_model", "Qwen/Qwen3-1.7B", str)

    def __init__(self):
        self._defaults: dict = {}
        self._data: dict = {}
        self._path = os.path.join(get_app_root(), "config", "settings.json")
        self._collect_defaults()
        self.load()

    def _collect_defaults(self):
        """Collect default values from all ConfigKey class attributes."""
        for attr_name in dir(self.__class__):
            attr = getattr(self.__class__, attr_name)
            if isinstance(attr, ConfigKey):
                self._defaults[attr.key] = attr.default

    def get(self, key: ConfigKey):
        val = self._data.get(key.key, key.default)
        if key.type_ is not None and val is not None:
            try:
                return key.type_(val)
            except (ValueError, TypeError):
                return key.default
        return val

    def set(self, key: ConfigKey, value):
        self._data[key.key] = value
        self.save()

    def reset(self):
        """Reset all settings to defaults."""
        self._data = dict(self._defaults)
        self.save()

    def load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
                return
            except (json.JSONDecodeError, OSError):
                pass

        # Attempt migration from old QSettings INI format
        old_path = os.path.join(get_app_root(), "config", "configv2.json")
        if os.path.exists(old_path):
            self._migrate_from_ini(old_path)
        else:
            self._data = dict(self._defaults)
        self.save()

    def _migrate_from_ini(self, ini_path: str):
        """Parse old qfluentwidgets QSettings INI config into self._data."""
        parser = configparser.ConfigParser()
        parser.read(ini_path, encoding='utf-8')
        self._data = dict(self._defaults)
        section = 'General'
        if parser.has_section(section):
            for key, value in parser.items(section):
                # Coerce bool strings
                if value.lower() == 'true':
                    self._data[key] = True
                elif value.lower() == 'false':
                    self._data[key] = False
                else:
                    # Try numeric, fall back to string
                    try:
                        self._data[key] = int(value)
                    except ValueError:
                        try:
                            self._data[key] = float(value)
                        except ValueError:
                            self._data[key] = value

    def save(self):
        """Atomic write to settings.json."""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp = self._path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, default=str)
        os.replace(tmp, self._path)


cfg = Config()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_config.py -v
```
Expected: All PASS

- [ ] **Step 5: Verify existing consumer files still import cleanly**

```bash
python -c "from src.config.config import cfg; print(cfg.get(cfg.engine))"
```
Expected: prints `XTTSv2` (or whatever is in settings.json)

- [ ] **Step 6: Commit**

```bash
git add src/config/config.py tests/test_config.py pytest.ini tests/__init__.py
git commit -m "feat: migrate config.py from qfluentwidgets to plain JSON backend"
```

---

### Task 3: AppState and AppCallbacks

**Spec:** `docs/superpowers/specs/2026-03-18-imgui-migration-design.md` §4, §10

**Files:**
- Create: `src/ui_imgui/__init__.py`
- Create: `src/ui_imgui/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_state.py
import queue
from src.ui_imgui.state import AppState, AppCallbacks

def test_appstate_initializes_with_defaults():
    state = AppState()
    assert state.loading is False
    assert state.loading_message == ""
    assert state.reference_audio == []
    assert state.recording_complete is False
    assert state.recording_file is None
    assert isinstance(state.error_queue, queue.Queue)
    assert isinstance(state.api_command_queue, queue.Queue)

def test_appcallbacks_on_progress_updates_state():
    state = AppState()
    cb = AppCallbacks.from_state(state)
    cb.on_progress("Loading model...")
    assert state.loading_message == "Loading model..."

def test_appcallbacks_on_done_clears_loading():
    state = AppState()
    state.loading = True
    cb = AppCallbacks.from_state(state)
    cb.on_done()
    assert state.loading is False

def test_appcallbacks_on_error_enqueues_message():
    state = AppState()
    cb = AppCallbacks.from_state(state)
    cb.on_error("Title", "Something went wrong")
    assert not state.error_queue.empty()
    title, msg = state.error_queue.get_nowait()
    assert title == "Title"
    assert msg == "Something went wrong"

def test_appcallbacks_on_media_update_sets_audio_file():
    state = AppState()
    cb = AppCallbacks.from_state(state)
    cb.on_media_update("/path/to/audio.wav")
    assert state.current_audio_file == "/path/to/audio.wav"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_state.py -v
```
Expected: All FAIL (ImportError)

- [ ] **Step 3: Create src/ui_imgui/__init__.py**

```python
```

- [ ] **Step 4: Create src/ui_imgui/state.py**

```python
from __future__ import annotations

import queue
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from src.tts_engines import tts_engine
    from src.enums.engine_type import EngineType


@dataclass
class AppCallbacks:
    """
    Replaces QMetaObject.invokeMethod cross-thread UI dispatch in src/utils/.
    Pass an instance of this as the `parent` argument to all utils functions
    (load_model, generic_inference, bulk_inference, etc.).
    """
    on_progress: Callable[[str], None]      # update loader message
    on_done: Callable[[], None]             # complete loader, clear loading flag
    on_error: Callable[[str, str], None]    # (title, message) → error queue
    on_warn: Callable[[str, str], None]     # (title, message) → warn queue
    on_model_loaded: Callable[[], None]     # after model load — enable widget
    on_media_update: Callable[[str], None]  # new output audio file path
    on_continue_load: Callable[[], None]    # after engine swap, resume pending op
    state_ref: Optional[AppState] = None   # read-only access for utils needing model/engine data

    @classmethod
    def from_state(cls, state: AppState) -> AppCallbacks:
        """Build a callbacks instance backed by the given AppState."""
        return cls(
            on_progress=lambda msg: setattr(state, 'loading_message', msg),
            on_done=lambda: (
                setattr(state, 'loading', False),
                setattr(state, 'loading_message', ''),
            ),
            on_error=lambda title, msg: state.error_queue.put((title, msg)),
            on_warn=lambda title, msg: state.warn_queue.put((title, msg)),
            on_model_loaded=lambda: setattr(state, 'model_just_loaded', True),
            on_media_update=lambda path: setattr(state, 'current_audio_file', path),
            on_continue_load=lambda: setattr(state, 'continue_load_pending', True),
            state_ref=state,
        )


@dataclass
class AppState:
    """Central shared mutable state. Passed by reference to every workspace/panel."""

    # Engine
    tts_engine: Optional[object] = None  # tts_engine instance
    engine_type: Optional[object] = None  # EngineType (set at runtime)

    # Models / characters
    characters_data: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    shared_models: Optional[list] = None
    custom_models: Optional[dict] = None
    default_references: Optional[dict] = None

    # Reference audio
    reference_audio: list = field(default_factory=list)
    reference_audio_length: float = 0.0

    # Pending loads (mirror FallTalkApp.pending_* fields)
    pending_character: Optional[str] = None
    pending_model: Optional[dict] = None
    pending_rvc: Optional[str] = None
    pending_base: bool = False
    pending_bulk: bool = False
    pending_ez: bool = False

    # Audio playback / recording
    current_audio_file: Optional[str] = None
    recording_file: Optional[str] = None
    recording_complete: bool = False    # set True by recorder thread; cleared by GenerationWorkspace after consuming

    # Loader
    loading: bool = False
    loading_message: str = ""

    # Message queues (thread-safe, drained by main loop each frame)
    error_queue: queue.Queue = field(default_factory=queue.Queue)
    warn_queue: queue.Queue = field(default_factory=queue.Queue)
    api_command_queue: queue.Queue = field(default_factory=queue.Queue)

    # Flags polled by workspaces each frame
    model_just_loaded: bool = False
    continue_load_pending: bool = False

    # Sub-engines
    upscale_engine: Optional[object] = None
    apbwe_engine: Optional[object] = None
    transcription_engine: Optional[object] = None

    # Callbacks (built at app init, passed to utils as parent-replacement)
    callbacks: Optional[AppCallbacks] = None

    # Engine-change subscribers (replaces Qt OnEngineChange signal)
    on_engine_change_listeners: list = field(default_factory=list)

    # Startup modal flags (polled by app.py show_gui on first frames)
    _show_disclaimer: bool = False
    _show_fallout_warning: bool = False
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_state.py -v
```
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/ui_imgui/__init__.py src/ui_imgui/state.py tests/test_state.py
git commit -m "feat: add AppState and AppCallbacks (replaces Qt cross-thread dispatch)"
```

---

### Task 4: Common widgets (AsyncWorker, loader overlay, error flyouts)

**Files:**
- Create: `src/ui_imgui/widgets/__init__.py`
- Create: `src/ui_imgui/widgets/common.py`

- [ ] **Step 1: Create src/ui_imgui/widgets/__init__.py**

```python
```

- [ ] **Step 2: Create src/ui_imgui/widgets/common.py**

```python
from __future__ import annotations

import threading
import time
from typing import Callable, Optional, TYPE_CHECKING

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


def draw_loader_overlay(state: AppState):
    """
    Render a full-viewport semi-transparent blocking overlay when state.loading is True.
    Call this at the end of every workspace's draw() method.
    """
    if not state.loading:
        return
    try:
        import imgui
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
        # Center the spinner text
        win_w, win_h = imgui.get_window_size()
        msg = state.loading_message or "Loading..."
        text_w = imgui.calc_text_size(msg).x
        imgui.set_cursor_pos(((win_w - text_w) * 0.5, win_h * 0.5))
        imgui.text(msg)
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
    import imgui
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
```

- [ ] **Step 3: Verify import**

```bash
python -c "from src.ui_imgui.widgets.common import AsyncWorker, draw_loader_overlay; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/ui_imgui/widgets/__init__.py src/ui_imgui/widgets/common.py
git commit -m "feat: add AsyncWorker, loader overlay, error flyouts"
```

---

### Task 5: File dialog helper

**Files:**
- Create: `src/ui_imgui/widgets/file_dialog.py`

- [ ] **Step 1: Create the file**

```python
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
```

- [ ] **Step 2: Verify import**

```bash
python -c "from src.ui_imgui.widgets.file_dialog import open_file; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/ui_imgui/widgets/file_dialog.py
git commit -m "feat: add tkinter-backed file/folder dialog helpers"
```

---

### Task 6: Workspace protocol + activity bar

**Files:**
- Create: `src/ui_imgui/workspace.py`
- Create: `src/ui_imgui/activity_bar.py`

- [ ] **Step 1: Create src/ui_imgui/workspace.py**

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Workspace:
    """
    Protocol that every workspace must implement.
    Adapted from I:\Fallout4Mods\Fallout4MCP\ui\toolkit\workspace.py.
    """

    @property
    def workspace_id(self) -> str:
        raise NotImplementedError

    @property
    def label(self) -> str:
        raise NotImplementedError

    @property
    def icon(self) -> str:
        """Icon label shown in activity bar (text emoji or short string)."""
        return "?"

    def get_dockable_windows(self) -> list:
        """Return list of hello_imgui.DockableWindow for this workspace."""
        return []

    def get_docking_splits(self) -> list:
        """Return list of hello_imgui.DockingSplit for initial layout."""
        return []

    def initialize(self):
        """Called once after GL context is available."""
        pass

    def draw(self):
        """Called every frame while this workspace is active."""
        pass

    def draw_menu(self):
        """Called every frame to add items to the main menu bar."""
        pass

    def on_activate(self):
        """Called when this workspace becomes active."""
        pass

    def on_deactivate(self):
        """Called when switching away from this workspace."""
        pass

    def cleanup(self):
        """Called on application exit."""
        pass
```

- [ ] **Step 2: Create src/ui_imgui/activity_bar.py**

```python
from __future__ import annotations
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.workspace import Workspace


class ActivityBar:
    """
    Left sidebar icon strip for workspace switching.
    Renders as a fixed-width imgui child window.
    """
    WIDTH = 48

    def __init__(self, workspaces: list[Workspace],
                 on_switch: Callable[[str], None]):
        self._workspaces = workspaces
        self._on_switch = on_switch
        self._active_id: Optional[str] = workspaces[0].workspace_id if workspaces else None

    @property
    def active_id(self) -> Optional[str]:
        return self._active_id

    def set_active(self, workspace_id: str):
        self._active_id = workspace_id

    def draw(self):
        import imgui
        vp = imgui.get_main_viewport()
        imgui.set_next_window_pos((vp.pos.x, vp.pos.y))
        imgui.set_next_window_size((self.WIDTH, vp.size.y))
        imgui.set_next_window_bg_alpha(1.0)
        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_scrollbar
        )
        imgui.begin("##activity_bar", flags=flags)
        for ws in self._workspaces:
            is_active = ws.workspace_id == self._active_id
            if is_active:
                imgui.push_style_color(imgui.Col_.button, (0.35, 0.50, 0.75, 1.0))
            if imgui.button(ws.icon, size=(36, 36)):
                if ws.workspace_id != self._active_id:
                    self._active_id = ws.workspace_id
                    self._on_switch(ws.workspace_id)
            if is_active:
                imgui.pop_style_color()
            if imgui.is_item_hovered():
                imgui.set_tooltip(ws.label)
            imgui.spacing()
        imgui.end()
```

- [ ] **Step 3: Verify imports**

```bash
python -c "from src.ui_imgui.workspace import Workspace; from src.ui_imgui.activity_bar import ActivityBar; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/ui_imgui/workspace.py src/ui_imgui/activity_bar.py
git commit -m "feat: add workspace protocol and activity bar"
```

---

### Task 7: Drawer overlay

**Files:**
- Create: `src/ui_imgui/widgets/drawer.py`

- [ ] **Step 1: Create src/ui_imgui/widgets/drawer.py**

```python
from __future__ import annotations
from typing import Callable, Optional


class Drawer:
    """
    Slide-in overlay panel pinned to the right edge of the viewport.
    Pass a callable to open(); it will be called every frame inside the drawer.
    """
    WIDTH = 350
    ANIM_SPEED = 8.0  # units per second (0→1)

    def __init__(self):
        self._open_pct: float = 0.0
        self._opening: bool = False
        self._content_fn: Optional[Callable] = None

    def open(self, content_fn: Callable):
        self._content_fn = content_fn
        self._opening = True

    def close(self):
        self._opening = False

    @property
    def is_open(self) -> bool:
        return self._open_pct > 0.0

    def draw(self):
        import imgui
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
        w = self.WIDTH * self._open_pct
        x = vp.pos.x + vp.size.x - w
        y = vp.pos.y
        h = vp.size.y

        imgui.set_next_window_pos((x, y))
        imgui.set_next_window_size((w, h))
        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
        )
        imgui.begin("##drawer", flags=flags)

        if self._open_pct >= 0.99:
            # Close button
            if imgui.button("X"):
                self.close()
            imgui.separator()
            if self._content_fn:
                self._content_fn()

        imgui.end()

        # Click outside to close
        if self._opening and imgui.is_mouse_clicked(0):
            mx, my = imgui.get_mouse_pos()
            if mx < x:
                self.close()
```

- [ ] **Step 2: Verify import**

```bash
python -c "from src.ui_imgui.widgets.drawer import Drawer; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/ui_imgui/widgets/drawer.py
git commit -m "feat: add animated drawer overlay"
```

---

### Task 8: Tables helper

**Files:**
- Create: `src/ui_imgui/widgets/tables.py`

- [ ] **Step 1: Create src/ui_imgui/widgets/tables.py**

```python
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
    import imgui
    if flags is None:
        flags = (
            imgui.TableFlags_.resizable
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.scroll_y
            | imgui.TableFlags_.sort_multi
        )
    if imgui.begin_table(table_id, len(columns), flags):
        for col in columns:
            imgui.table_setup_column(col)
        imgui.table_headers_row()
        return True
    return False


def end_table():
    import imgui
    imgui.end_table()


def search_filter(filter_id: str, filter_text: list[str], width: float = 200.0) -> bool:
    """
    Render a search input. filter_text is a 1-element list (mutable string).
    Returns True if the value changed.
    """
    import imgui
    imgui.set_next_item_width(width)
    changed, new_val = imgui.input_text(f"Search##{filter_id}", filter_text[0])
    if changed:
        filter_text[0] = new_val
    return changed
```

- [ ] **Step 2: Verify import**

```bash
python -c "from src.ui_imgui.widgets.tables import begin_table; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/ui_imgui/widgets/tables.py
git commit -m "feat: add reusable imgui table helpers"
```

---

### Task 9: hello_imgui bootstrap (app.py)

**Files:**
- Create: `src/ui_imgui/app.py`
- Create: `src/ui_imgui/shared_panels.py`

This is the main entry point for the new UI. It wires everything together.

- [ ] **Step 1: Create src/ui_imgui/shared_panels.py (stub)**

```python
"""Persistent panels that survive workspace switches."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState


class SharedPanels:
    """Container for panels that stay alive across workspace switches."""

    def __init__(self, state: AppState):
        self._state = state
        # AudioPanel is imported lazily to avoid circular imports
        self._audio_panel = None

    def initialize(self):
        from src.ui_imgui.widgets.audio_panel import AudioPanel
        self._audio_panel = AudioPanel(self._state)

    def get_dockable_windows(self) -> list:
        from imgui_bundle import hello_imgui
        if self._audio_panel is None:
            return []
        w = hello_imgui.DockableWindow()
        w.label = "Audio##shared"
        w.dock_space_name = "BottomSpace"
        w.call_begin_end = True
        w.gui_function = self._audio_panel.draw
        return [w]
```

- [ ] **Step 2: Create src/ui_imgui/app.py**

```python
from __future__ import annotations

import json
import os
import queue
import sys

from imgui_bundle import hello_imgui, imgui, immapp

from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.ui_imgui.activity_bar import ActivityBar
from src.ui_imgui.shared_panels import SharedPanels
from src.ui_imgui.state import AppState, AppCallbacks
from src.ui_imgui.widgets.common import drain_error_queue, draw_error_flyouts, draw_loader_overlay
from src.utils.filesystem_utils import get_app_root


def _apply_darcula_theme():
    """Dark Darcula-inspired theme (mirrors reference implementation)."""
    style = imgui.get_style()
    colors = style.colors
    colors[imgui.Col_.window_bg.value] = (0.12, 0.12, 0.14, 1.0)
    colors[imgui.Col_.child_bg.value] = (0.10, 0.10, 0.12, 1.0)
    colors[imgui.Col_.popup_bg.value] = (0.14, 0.14, 0.16, 1.0)
    colors[imgui.Col_.frame_bg.value] = (0.18, 0.18, 0.20, 1.0)
    colors[imgui.Col_.frame_bg_hovered.value] = (0.22, 0.22, 0.25, 1.0)
    colors[imgui.Col_.frame_bg_active.value] = (0.25, 0.25, 0.28, 1.0)
    colors[imgui.Col_.title_bg.value] = (0.10, 0.10, 0.12, 1.0)
    colors[imgui.Col_.title_bg_active.value] = (0.15, 0.15, 0.18, 1.0)
    colors[imgui.Col_.button.value] = (0.22, 0.22, 0.25, 1.0)
    colors[imgui.Col_.button_hovered.value] = (0.30, 0.30, 0.35, 1.0)
    colors[imgui.Col_.button_active.value] = (0.35, 0.50, 0.75, 1.0)
    colors[imgui.Col_.header.value] = (0.22, 0.22, 0.25, 1.0)
    colors[imgui.Col_.header_hovered.value] = (0.30, 0.45, 0.70, 1.0)
    colors[imgui.Col_.header_active.value] = (0.35, 0.50, 0.75, 1.0)
    colors[imgui.Col_.separator.value] = (0.28, 0.28, 0.30, 1.0)
    colors[imgui.Col_.text.value] = (0.85, 0.85, 0.85, 1.0)
    colors[imgui.Col_.check_mark.value] = (0.40, 0.70, 1.0, 1.0)
    colors[imgui.Col_.slider_grab.value] = (0.40, 0.60, 0.90, 1.0)
    colors[imgui.Col_.slider_grab_active.value] = (0.50, 0.70, 1.0, 1.0)
    style.window_rounding = 4.0
    style.frame_rounding = 2.0
    style.grab_rounding = 2.0
    style.scrollbar_rounding = 2.0


def _build_workspaces(state: AppState) -> list:
    """Import and instantiate all workspaces."""
    from src.ui_imgui.workspaces.characters import CharactersWorkspace
    from src.ui_imgui.workspaces.generation import GenerationWorkspace
    from src.ui_imgui.workspaces.bulk import BulkWorkspace
    from src.ui_imgui.workspaces.chat import ChatWorkspace
    from src.ui_imgui.workspaces.upscale import UpscaleWorkspace
    from src.ui_imgui.workspaces.settings import SettingsWorkspace
    return [
        CharactersWorkspace(state),
        GenerationWorkspace(state),
        BulkWorkspace(state),
        ChatWorkspace(state),
        UpscaleWorkspace(state),
        SettingsWorkspace(state),
    ]


def run():
    """Entry point for the imgui FallTalk app."""
    # ── Check api_only_mode ──────────────────────────────────────────────────
    if cfg.get(cfg.api_only_mode):
        _run_api_only()
        return

    # ── Build shared state ───────────────────────────────────────────────────
    state = AppState()
    state.callbacks = AppCallbacks.from_state(state)
    state.engine_type = EngineType(cfg.get(cfg.engine))

    # ── Load static JSON data ────────────────────────────────────────────────
    _load_static_data(state)

    # ── Build workspaces ─────────────────────────────────────────────────────
    workspaces = _build_workspaces(state)
    shared = SharedPanels(state)

    active_ws_id: list[str] = [workspaces[0].workspace_id]

    def switch_workspace(ws_id: str):
        for ws in workspaces:
            if ws.workspace_id == ws_id:
                ws.on_activate()
            elif ws.workspace_id == active_ws_id[0]:
                ws.on_deactivate()
        active_ws_id[0] = ws_id

    activity_bar = ActivityBar(workspaces, switch_workspace)

    # ── hello_imgui RunnerParams ─────────────────────────────────────────────
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = f"FallTalk"
    runner_params.app_window_params.window_geometry.size = (1280, 900)
    runner_params.imgui_window_params.show_menu_bar = True
    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    )

    # Docking layout (built from first workspace on startup)
    docking_params = hello_imgui.DockingParams()
    for ws in workspaces:
        docking_params.dockable_windows += ws.get_dockable_windows()
        docking_params.docking_splits += ws.get_docking_splits()
    docking_params.dockable_windows += shared.get_dockable_windows()
    runner_params.docking_params = docking_params

    _theme_applied = [False]

    def post_init():
        shared.initialize()
        for ws in workspaces:
            ws.initialize()
        workspaces[0].on_activate()
        _startup_checks(state)

    def show_gui():
        if not _theme_applied[0]:
            _apply_darcula_theme()
            _theme_applied[0] = True

        # Draw activity bar
        activity_bar.draw()

        # Drain queues
        drain_error_queue(state)

        # Draw active workspace menu
        for ws in workspaces:
            if ws.workspace_id == active_ws_id[0]:
                ws.draw_menu()

        # Drain API command queue
        _drain_api_queue(state)

        # Draw overlay widgets (error flyouts, loader)
        draw_error_flyouts()
        draw_loader_overlay(state)

        # Poll model_just_loaded flag
        if state.model_just_loaded:
            state.model_just_loaded = False
            _on_model_loaded(state, workspaces)

    runner_params.callbacks.post_init = post_init
    runner_params.callbacks.show_gui = show_gui

    addons = immapp.AddOnsParams()
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
    # These open imgui modals on the first frame via deferred open_popup
    if not cfg.get(cfg.accepted_disclaimer):
        state._show_disclaimer = True
    if cfg.get(cfg.fallout_4_directory_check):
        if cfg.get(cfg.fallout_4_directory) == "fallout4.exe not found":
            state._show_fallout_warning = True
    if cfg.get(cfg.check_for_updates):
        from src.ui_imgui.widgets.common import AsyncWorker
        from src.utils.huggingface_utils import get_latest_release
        AsyncWorker(get_latest_release).start()


def _on_model_loaded(state: AppState, workspaces: list):
    """Handle model_just_loaded flag — notify all workspaces."""
    for ws in workspaces:
        if hasattr(ws, 'on_model_loaded'):
            ws.on_model_loaded()


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
    from src.api.falltalkapi import FallTalkAPI
    import uvicorn
    api = FallTalkAPI(None)
    uvicorn.run(api.app, host="0.0.0.0", port=8000)
```

- [ ] **Step 3: Verify import (no display needed)**

```bash
python -c "from src.ui_imgui.app import run, _apply_darcula_theme; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/ui_imgui/app.py src/ui_imgui/shared_panels.py
git commit -m "feat: add hello_imgui bootstrap app.py with Darcula theme and startup flow"
```

> **Phase 1 complete.** Streams 2–6 may now begin in parallel.

---

## Phase 2: Parallel Streams

> All streams in this phase are independent. Assign to separate agents and run simultaneously.
> Each stream should read: `docs/superpowers/specs/2026-03-18-imgui-migration-design.md` for full detail.
> Every stream should import `AppState` from `src.ui_imgui.state` and use `cfg` from `src.config.config`.

---

### Stream 2: Characters & References Workspace

**Files:**
- Create: `src/ui_imgui/workspaces/__init__.py`
- Create: `src/ui_imgui/workspaces/characters.py`

**Implements:** CharactersWorkspace with two dockable panels.

---

#### Task 10: Characters workspace scaffold

- [ ] **Step 1: Create src/ui_imgui/workspaces/__init__.py**

```python
```

- [ ] **Step 2: Create src/ui_imgui/workspaces/characters.py scaffold**

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState

from src.ui_imgui.workspace import Workspace


class CharactersWorkspace(Workspace):
    workspace_id = "characters"
    label = "Characters"
    icon = "CC"  # replace with icon font glyph when available

    def __init__(self, state: AppState):
        self._state = state
        self._char_filter = [""]      # mutable search string
        self._ref_filter = [""]
        self._active_tab = 0          # 0=Trained, 1=Untrained, 2=Custom

    def get_docking_splits(self) -> list:
        from imgui_bundle import hello_imgui
        split = hello_imgui.DockingSplit()
        split.initial_dock = "MainDockSpace"
        split.new_dock = "RightSpace"
        split.direction = imgui_bundle.imgui.Dir_.right
        split.ratio = 0.45
        return [split]

    def get_dockable_windows(self) -> list:
        from imgui_bundle import hello_imgui
        wins = []
        w1 = hello_imgui.DockableWindow()
        w1.label = "Characters##ws"
        w1.dock_space_name = "MainDockSpace"
        w1.call_begin_end = True
        w1.gui_function = self._draw_characters_panel
        wins.append(w1)
        w2 = hello_imgui.DockableWindow()
        w2.label = "References##ws"
        w2.dock_space_name = "RightSpace"
        w2.call_begin_end = True
        w2.gui_function = self._draw_references_panel
        wins.append(w2)
        return wins

    def _draw_characters_panel(self):
        """Characters panel: tabs for Trained / Untrained / Custom models."""
        from imgui_bundle import imgui
        from src.ui_imgui.widgets.tables import search_filter, begin_table, end_table
        from src.config.config import cfg

        search_filter("char_search", self._char_filter)
        imgui.same_line()
        if imgui.button("Import Custom Model"):
            self._show_import_dialog()

        imgui.separator()

        if imgui.begin_tab_bar("char_tabs"):
            if imgui.begin_tab_item("Trained")[0]:
                self._draw_character_table("trained", self._get_trained_characters())
                imgui.end_tab_item()
            if imgui.begin_tab_item("Untrained")[0]:
                self._draw_character_table("untrained", self._get_untrained_characters())
                imgui.end_tab_item()
            if imgui.begin_tab_item("Custom")[0]:
                self._draw_character_table("custom", self._get_custom_characters())
                imgui.end_tab_item()
            imgui.end_tab_bar()

    def _draw_character_table(self, table_id: str, characters: list):
        from imgui_bundle import imgui
        from src.ui_imgui.widgets.tables import begin_table, end_table
        cols = ["Name", "Load", "Download", "Update", "Delete", "RVC"]
        if begin_table(f"char_table_{table_id}", cols):
            flt = self._char_filter[0].lower()
            for char in characters:
                name = char.get('display_name', char.get('name', ''))
                if flt and flt not in name.lower():
                    continue
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                imgui.text(name)
                imgui.table_set_column_index(1)
                if imgui.small_button(f"Load##{name}"):
                    self._load_character(char)
                imgui.table_set_column_index(2)
                if imgui.small_button(f"DL##{name}"):
                    self._download_character(char)
                imgui.table_set_column_index(3)
                if imgui.small_button(f"Upd##{name}"):
                    self._update_character(char)
                imgui.table_set_column_index(4)
                if imgui.small_button(f"Del##{name}"):
                    self._delete_character(char)
                imgui.table_set_column_index(5)
                has_rvc = char.get('RVC') is not None
                imgui.text("Yes" if has_rvc else "No")
            end_table()

    def _draw_references_panel(self):
        """References panel: audio table with search."""
        from imgui_bundle import imgui
        from src.ui_imgui.widgets.tables import search_filter, begin_table, end_table

        search_filter("ref_search", self._ref_filter)
        imgui.separator()

        cols = ["Filename", "Dialogue", "Plugin", "Folder", "Duration"]
        if begin_table("ref_table", cols):
            # References are loaded from self._state.reference_audio
            # plus BSA/XWM extraction results
            for ref in self._state.reference_audio:
                fname = os.path.basename(ref) if isinstance(ref, str) else str(ref)
                if self._ref_filter[0] and self._ref_filter[0].lower() not in fname.lower():
                    continue
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                imgui.text(fname)
                # Remaining columns populated when reference data includes metadata
            end_table()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _get_trained_characters(self) -> list:
        from src.config.config import cfg
        result = []
        for name, char in self._state.characters_data.items():
            if name in self._state.models and cfg.get(cfg.engine) in self._state.models[name]:
                c = dict(char)
                c['display_name'] = self._state.models[name].get('display_name', name)
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
                result.append(c)
        return result

    def _get_custom_characters(self) -> list:
        if not self._state.custom_models:
            return []
        return list(self._state.custom_models.values())

    def _load_character(self, char: dict):
        import threading
        from src.utils.model_utils import load_model
        from src.config.config import cfg
        from src.enums.engine_type import EngineType
        engine_key = cfg.get(cfg.engine)
        model = self._state.models.get(char['name'], {}).get(engine_key)
        rvc = self._state.models.get(char['name'], {}).get('RVC')
        self._state.loading = True
        self._state.loading_message = f"Loading {char.get('display_name', char['name'])}..."
        cb = self._state.callbacks
        threading.Thread(
            target=load_model,
            args=(cb, char['name'], rvc, char.get('display_name', char['name']), False),
            daemon=True
        ).start()

    def _download_character(self, char: dict):
        import threading
        from src.utils.huggingface_utils import download_models
        model = self._state.models.get(char['name'], {})
        rvc = model.get('RVC')
        self._state.loading = True
        self._state.loading_message = f"Downloading {char.get('display_name', char['name'])}..."
        threading.Thread(
            target=download_models,
            args=(self._state.callbacks, char['name'], model, rvc),
            daemon=True
        ).start()

    def _update_character(self, char: dict):
        # Same as download but deletes existing first
        self._download_character(char)

    def _delete_character(self, char: dict):
        # Show confirmation (imgui modal), then delete
        pass  # TODO: add confirmation modal

    def _show_import_dialog(self):
        from src.ui_imgui.widgets.file_dialog import open_file
        path = open_file("Import Model", [("Model files", "*.pth *.ckpt *.index")])
        if path:
            pass  # TODO: handle import


import os
```

- [ ] **Step 3: Commit scaffold**

```bash
git add src/ui_imgui/workspaces/__init__.py src/ui_imgui/workspaces/characters.py
git commit -m "feat: add Characters workspace scaffold with panels"
```

- [ ] **Step 4: Complete _draw_references_panel with full reference data display**

The references panel should:
- Show `filename`, `dialogue`, `arcname`, `plugin`, `folder`, `duration` columns
- Support multi-select (checkboxes)
- Show total selected duration
- Play selected reference in AudioPanel (set `state.current_audio_file`)
- Populate from `self._state.reference_audio` metadata if available, else just filenames

- [ ] **Step 5: Wire character click → reference panel update**

When a character is loaded in the Characters panel, call `reference_widget.addDataToReferencesTable` equivalent — i.e., populate `state.reference_audio` with that character's reference files from the `config/` data.

- [ ] **Step 6: Commit completed workspace**

```bash
git add src/ui_imgui/workspaces/characters.py
git commit -m "feat: complete Characters + References workspace"
```

---

### Stream 3: Generation Workspace + Engine Settings/Help

**Files:**
- Create: `src/ui_imgui/workspaces/generation.py`
- Create: `src/ui_imgui/widgets/engine_settings.py`
- Create: `src/ui_imgui/widgets/engine_help.py`

---

#### Task 11: Engine settings consolidation (replaces 20 settings files)

- [ ] **Step 1: Create src/ui_imgui/widgets/engine_settings.py**

Implement `draw_engine_settings(state: AppState)` with a `match state.engine_type` block. Each case renders imgui sliders/checkboxes/combos for that engine's config keys. Reference `src/settings/<engine>_settings.py` for the full parameter list per engine.

Example structure:
```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.ui_imgui.state import AppState
from src.config.config import cfg
from src.enums.engine_type import EngineType


def draw_engine_settings(state: AppState):
    """Render engine-specific settings. Called inside the drawer or Settings panel."""
    from imgui_bundle import imgui
    if state.engine_type is None:
        imgui.text("No engine loaded.")
        return

    match state.engine_type:
        case EngineType.XTTS_V2:
            _draw_xtts_settings()
        case EngineType.GPT_SOVITS:
            _draw_gpt_sovits_settings()
        case EngineType.F5:
            _draw_f5_settings()
        case EngineType.FISH_SPEECH:
            _draw_fish_settings()
        case EngineType.STYLE_TTS2:
            _draw_styletts2_settings()
        case EngineType.DIA:
            _draw_dia_settings()
        case EngineType.LLASA:
            _draw_llasa_settings()
        case EngineType.ORPHEUS:
            _draw_orpheus_settings()
        case EngineType.SPARK:
            _draw_spark_settings()
        case EngineType.CSM:
            _draw_csm_settings()
        case EngineType.HIGGS:
            _draw_higgs_settings()
        case EngineType.CHATTERBOX:
            _draw_chatterbox_settings()
        case EngineType.DMOSPEECH2:
            _draw_dmo_settings()
        case EngineType.VIBE:
            _draw_vibe_settings()
        case EngineType.QWEN3_TTS:
            _draw_qwen_settings()
        case EngineType.RVC:
            _draw_rvc_settings()
        case _:
            imgui.text(f"No settings for {state.engine_type}")


def _draw_xtts_settings():
    from imgui_bundle import imgui
    changed, val = imgui.slider_int("Speed##xtts", cfg.get(cfg.speed), 1, 200)
    if changed: cfg.set(cfg.speed, val)
    changed, val = imgui.slider_int("Temperature##xtts", cfg.get(cfg.model_temperature), 1, 100)
    if changed: cfg.set(cfg.model_temperature, val)
    changed, val = imgui.slider_int("Repetition##xtts", cfg.get(cfg.model_repetition), 1, 15)
    if changed: cfg.set(cfg.model_repetition, val)
    changed, val = imgui.checkbox("Low VRAM##xtts", cfg.get(cfg.low_vram))
    if changed: cfg.set(cfg.low_vram, val)
    changed, val = imgui.checkbox("DeepSpeed##xtts", cfg.get(cfg.deepspeed_enabled))
    if changed: cfg.set(cfg.deepspeed_enabled, val)

# Implement all remaining _draw_<engine>_settings() functions following the same
# pattern. Refer to src/settings/<engine>_settings.py for parameter names and ranges.
```

- [ ] **Step 2: Implement all engine settings functions** (one per engine listed in `src/settings/`)

  For each engine, read the parameter list and ranges from the corresponding `src/settings/<engine>_settings.py`. Implement with appropriate imgui widgets:
  - int/float sliders → `imgui.slider_int` / `imgui.slider_float`
  - bool → `imgui.checkbox`
  - enum/choice → `imgui.combo` with list of options

- [ ] **Step 3: Commit**

```bash
git add src/ui_imgui/widgets/engine_settings.py
git commit -m "feat: add engine_settings.py consolidating all 20 settings files"
```

---

#### Task 12: Engine help consolidation (replaces 20+ help files)

- [ ] **Step 1: Create src/ui_imgui/widgets/engine_help.py**

```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.ui_imgui.state import AppState
from src.enums.engine_type import EngineType

# ── Help text constants ──────────────────────────────────────────────────────
XTTS_HELP = """
XTTSv2 — Coqui TTS cross-lingual model.
Requires 3–10 seconds of reference audio.
Speed: 1=normal, 200=fast. Temperature: higher = more expressive.
"""
# (add one constant per engine, sourced from src/help/<engine>_help.py)


def draw_engine_help(state: AppState):
    """Render engine-specific help content."""
    from imgui_bundle import imgui
    if state.engine_type is None:
        imgui.text("Load an engine to see help.")
        return

    match state.engine_type:
        case EngineType.XTTS_V2:
            imgui.text_wrapped(XTTS_HELP)
        # Add all other engines...
        case _:
            imgui.text(f"No help available for {state.engine_type}")
```

- [ ] **Step 2: Populate all help text constants** from `src/help/<engine>_help.py`

- [ ] **Step 3: Commit**

```bash
git add src/ui_imgui/widgets/engine_help.py
git commit -m "feat: add engine_help.py consolidating all 20+ help files"
```

---

#### Task 13: Generation workspace

- [ ] **Step 1: Create src/ui_imgui/workspaces/generation.py**

```python
from __future__ import annotations
import threading
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.ui_imgui.state import AppState

from src.ui_imgui.workspace import Workspace
from src.ui_imgui.widgets.drawer import Drawer
from src.config.config import cfg
from src.enums.engine_type import EngineType


class GenerationWorkspace(Workspace):
    workspace_id = "generation"
    label = "Generation"
    icon = "GN"

    def __init__(self, state: AppState):
        self._state = state
        self._drawer = Drawer()
        self._text_input = [""]            # mutable text buffer
        self._output_name = [""]
        self._multigen_results: list[str] = []

    def get_docking_splits(self) -> list:
        from imgui_bundle import hello_imgui, imgui
        splits = []
        s1 = hello_imgui.DockingSplit()
        s1.initial_dock = "MainDockSpace"
        s1.new_dock = "RightGenSpace"
        s1.direction = imgui.Dir_.right
        s1.ratio = 0.35
        splits.append(s1)
        s2 = hello_imgui.DockingSplit()
        s2.initial_dock = "MainDockSpace"
        s2.new_dock = "BottomGenSpace"
        s2.direction = imgui.Dir_.down
        s2.ratio = 0.30
        splits.append(s2)
        return splits

    def get_dockable_windows(self) -> list:
        from imgui_bundle import hello_imgui
        wins = []
        for label, dock, fn in [
            ("Generation##gen", "MainDockSpace", self._draw_generation_panel),
            ("Engine Settings##gen", "RightGenSpace", self._draw_settings_panel),
            ("Help##gen", "RightGenSpace", self._draw_help_panel),
            ("Multi-Generation##gen", "BottomGenSpace", self._draw_multigen_panel),
        ]:
            w = hello_imgui.DockableWindow()
            w.label = label
            w.dock_space_name = dock
            w.call_begin_end = True
            w.gui_function = fn
            wins.append(w)
        return wins

    def draw(self):
        # Poll recording_complete flag
        if self._state.recording_complete:
            self._state.recording_complete = False
            self._start_rvc_from_recording()

    def _draw_generation_panel(self):
        from imgui_bundle import imgui
        from src.utils.inference_utils import preprocess_text

        engine_type = self._state.engine_type
        engine_loaded = self._state.tts_engine is not None

        # Text input
        imgui.text("Text to Generate:")
        imgui.set_next_item_width(-1)
        _, self._text_input[0] = imgui.input_text_multiline(
            "##text_input", self._text_input[0], size=(0, 120)
        )

        # Output name
        imgui.text("Output Name:")
        imgui.set_next_item_width(200)
        _, self._output_name[0] = imgui.input_text("##out_name", self._output_name[0])

        imgui.same_line()

        # RVC / Upscaler toggles
        changed, val = imgui.checkbox("RVC", cfg.get(cfg.rvc_enabled))
        if changed: cfg.set(cfg.rvc_enabled, val)
        imgui.same_line()
        changed, val = imgui.checkbox("Enhance", cfg.get(cfg.apbwe_enabled))
        if changed: cfg.set(cfg.apbwe_enabled, val)

        imgui.separator()

        # Generate button
        if not engine_loaded:
            imgui.text_colored((0.7, 0.7, 0.7, 1.0), "Load a model to generate.")
        else:
            if imgui.button("Generate", size=(120, 0)):
                self._start_generation()

        imgui.same_line()
        if imgui.button("Settings"):
            from src.ui_imgui.widgets.engine_settings import draw_engine_settings
            self._drawer.open(lambda: draw_engine_settings(self._state))
        imgui.same_line()
        if imgui.button("Help"):
            from src.ui_imgui.widgets.engine_help import draw_engine_help
            self._drawer.open(lambda: draw_engine_help(self._state))

        self._drawer.draw()

    def _draw_settings_panel(self):
        from src.ui_imgui.widgets.engine_settings import draw_engine_settings
        draw_engine_settings(self._state)

    def _draw_help_panel(self):
        from src.ui_imgui.widgets.engine_help import draw_engine_help
        draw_engine_help(self._state)

    def _draw_multigen_panel(self):
        from imgui_bundle import imgui
        imgui.text(f"Multi-Generation Results ({len(self._multigen_results)})")
        imgui.separator()
        for i, path in enumerate(self._multigen_results):
            import os
            imgui.text(os.path.basename(path))
            imgui.same_line()
            if imgui.small_button(f"Play##{i}"):
                self._state.current_audio_file = path
            imgui.same_line()
            if imgui.small_button(f"Save##{i}"):
                pass  # copy to output dir

    def _start_generation(self):
        text = self._text_input[0].strip()
        if not text:
            self._state.error_queue.put(("Error", "Please enter text to generate."))
            return
        references = self._state.reference_audio
        from src.utils.inference_utils import generic_inference, preprocess_text, combine_references
        from src.utils.file_utils import get_output_file_name
        output_file = get_output_file_name(
            self._output_name[0] or "output",
            cfg.get(cfg.output_dir),
            self._state.tts_engine.model_name if self._state.tts_engine else "unknown",
            self._state.engine_type.value if self._state.engine_type else "unknown"
        )
        self._state.loading = True
        self._state.loading_message = "Generating audio..."
        threading.Thread(
            target=generic_inference,
            args=(
                self._state.callbacks,
                output_file,
                preprocess_text(text),
                combine_references(references),
                None,  # widget (not needed in imgui)
                None,  # transcribe_state
            ),
            kwargs={'speaker': self._state.tts_engine.model_name if self._state.tts_engine else None, 'api': False},
            daemon=True
        ).start()

    def _start_rvc_from_recording(self):
        recording = self._state.recording_file
        if not recording:
            return
        from src.utils.inference_utils import rvc_inference
        from src.utils.file_utils import get_output_file_name
        output_file = get_output_file_name(
            "rvc_recording", cfg.get(cfg.output_dir), "rvc", "RVC"
        )
        import shutil
        shutil.copy(recording, output_file)
        self._state.loading = True
        self._state.loading_message = "Converting with RVC..."
        threading.Thread(
            target=rvc_inference,
            args=(self._state.callbacks, output_file, None),
            daemon=True
        ).start()
```

- [ ] **Step 2: Commit**

```bash
git add src/ui_imgui/workspaces/generation.py
git commit -m "feat: add Generation workspace with text input, settings/help panels, multi-gen"
```

---

### Stream 4: Bulk & EzVoice Workspace

**Files:**
- Create: `src/ui_imgui/workspaces/bulk.py`

---

#### Task 14: Bulk workspace

- [ ] **Step 1: Create src/ui_imgui/workspaces/bulk.py**

Implement `BulkWorkspace(Workspace)` with four dockable panels:

1. **Bulk CSV Panel** — file picker for CSV, editable table (`imgui.input_text` per cell), generate button, progress display
2. **Bulk RVC Panel** — folder picker via `file_dialog.open_folder()`, character combo (from `state.models`), thread count slider, include-subdirs checkbox
3. **Bulk FUZ Panel** — folder picker, include-subdirs checkbox, generate button
4. **EzVoice Panel** — table with columns `FILE_NAME, RESPONSE_TEXT, VOICE_TYPE, FULLPATH, REFERENCE_FILE, PLUGIN`, CSV import button, generate button

All generate buttons use `AsyncWorker` + `state.callbacks`. Progress updates via `state.callbacks.on_progress`.

Reference: `src/widgets/bulk_generation_widgets.py` for feature parity.

- [ ] **Step 2: Commit**

```bash
git add src/ui_imgui/workspaces/bulk.py
git commit -m "feat: add Bulk workspace (CSV, RVC, FUZ, EzVoice panels)"
```

---

### Stream 5: Chat, Upscale, Settings Workspaces

**Files:**
- Create: `src/ui_imgui/workspaces/chat.py`
- Create: `src/ui_imgui/workspaces/upscale.py`
- Create: `src/ui_imgui/workspaces/settings.py`

---

#### Task 15: Chat workspace

- [ ] **Step 1: Create src/ui_imgui/workspaces/chat.py**

Implement `ChatWorkspace(Workspace)` with:
- Scrollable chat history (list of `(role, text, audio_path)` tuples)
- Text input at bottom
- Send button → LLM inference via existing `chat` utils
- Per-message play button → `state.current_audio_file = audio_path`
- Clear chat button

Reference: `src/widgets/chat_widget.py`

- [ ] **Step 2: Commit**

```bash
git add src/ui_imgui/workspaces/chat.py
git commit -m "feat: add Chat workspace"
```

---

#### Task 16: Upscale workspace

- [ ] **Step 1: Create src/ui_imgui/workspaces/upscale.py**

Implement `UpscaleWorkspace(Workspace)` with:
- Mode radio buttons: Denoise / Isolate Vocals / Upscale
- Folder picker (via `file_dialog.open_folder()`)
- Sample rate combo: `44100` / `48000`
- Include subdirs checkbox
- Replace existing checkbox
- Generate button → `load_upscaler` then `upscale_engine.upscale_dir`

Reference: `src/widgets/upscale_widget.py`

- [ ] **Step 2: Commit**

```bash
git add src/ui_imgui/workspaces/upscale.py
git commit -m "feat: add Upscale workspace"
```

---

#### Task 17: Settings workspace

- [ ] **Step 1: Create src/ui_imgui/workspaces/settings.py**

Implement `SettingsWorkspace(Workspace)` with two panels:

**Main Settings Panel** — categories as `imgui.collapsing_header`:
- General: engine combo, device radio (CPU/CUDA/CUDA:1), auto-play checkbox, DPI scale (`font_global_scale`), check-for-updates, download-configs
- Paths: output dir, fallout4 dir, huggingface cache dir (all via `file_dialog.open_folder()`)
- Text Processing: max text size, min chunk size, all bool toggles
- Bulk: replace existing, include subdirs, threads, multigen count
- Audio: output device combo, input device combo (from `sounddevice.query_devices()`)
- API: api only mode toggle, disable SSL verify
- Reset to Defaults button → `cfg.reset()`

**FAQ Panel** — `imgui.collapsing_header` sections with text from `src/widgets/faq.py` and `src/widgets/faq_widget.py`. Include disclaimer text.

- [ ] **Step 2: Commit**

```bash
git add src/ui_imgui/workspaces/settings.py
git commit -m "feat: add Settings workspace with global config + FAQ panels"
```

---

### Stream 6: Audio Panel

> Depends on `state.recording_file` and `state.recording_complete` being defined in Stream 1 (`state.py`). Confirm these fields exist before starting.

**Files:**
- Create: `src/ui_imgui/widgets/audio_panel.py`

---

#### Task 18: Audio panel

- [ ] **Step 1: Create src/ui_imgui/widgets/audio_panel.py**

```python
from __future__ import annotations
import threading
import time
import numpy as np
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from src.ui_imgui.state import AppState


class AudioPanel:
    """
    Persistent dockable audio player + waveform display + RVC recorder.
    Observes state.current_audio_file each frame.
    """

    def __init__(self, state: AppState):
        self._state = state
        self._loaded_file: Optional[str] = None
        self._samples: Optional[np.ndarray] = None
        self._sample_rate: int = 44100
        self._duration: float = 0.0
        self._play_pos: float = 0.0
        self._playing: bool = False
        self._play_start_time: float = 0.0
        self._recording: bool = False
        self._rec_start_time: float = 0.0
        self._rec_stream = None
        self._rec_frames: list = []
        self._output_devices: list[str] = []
        self._input_devices: list[str] = []
        self._out_device_idx: int = 0
        self._in_device_idx: int = 0
        self._enumerate_devices()

    def _enumerate_devices(self):
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            self._output_devices = [d['name'] for d in devices if d['max_output_channels'] > 0]
            self._input_devices = [d['name'] for d in devices if d['max_input_channels'] > 0]
        except Exception:
            self._output_devices = ["Default"]
            self._input_devices = ["Default"]

    def _load_audio(self, path: str):
        try:
            import soundfile as sf
            data, sr = sf.read(path, always_2d=False)
            if data.ndim > 1:
                data = data[:, 0]  # mono
            # Downsample to ~2000 points for waveform display
            step = max(1, len(data) // 2000)
            self._samples = data[::step].astype(np.float32)
            self._sample_rate = sr
            self._duration = len(data) / sr
            self._loaded_file = path
            self._play_pos = 0.0
        except Exception:
            self._samples = None
            self._duration = 0.0

    def _play(self):
        if self._loaded_file is None:
            return
        try:
            import sounddevice as sd
            import soundfile as sf
            data, sr = sf.read(self._loaded_file, always_2d=False)
            dev_idx = self._out_device_idx if self._out_device_idx < len(self._output_devices) else None
            self._playing = True
            self._play_start_time = time.time()
            sd.play(data, sr, device=dev_idx)
        except Exception:
            self._playing = False

    def _stop(self):
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self._playing = False
        self._play_pos = 0.0

    def _start_recording(self):
        try:
            import sounddevice as sd
            self._rec_frames = []
            self._rec_start_time = time.time()
            self._recording = True
            dev_idx = self._in_device_idx if self._in_device_idx < len(self._input_devices) else None

            def callback(indata, frames, t, status):
                self._rec_frames.append(indata.copy())

            self._rec_stream = sd.InputStream(
                samplerate=40000, channels=1, dtype='float32',
                device=dev_idx, callback=callback
            )
            self._rec_stream.start()
        except Exception:
            self._recording = False

    def _stop_recording(self):
        if self._rec_stream:
            self._rec_stream.stop()
            self._rec_stream.close()
            self._rec_stream = None
        self._recording = False
        if self._rec_frames:
            self._save_recording()

    def _save_recording(self):
        import soundfile as sf
        import os
        from src.utils.file_utils import formatted_time_stamp_uuid
        path = os.path.join("temp", f"recording_{formatted_time_stamp_uuid()}.wav")
        os.makedirs("temp", exist_ok=True)
        data = np.concatenate(self._rec_frames, axis=0)
        sf.write(path, data, 40000)
        self._state.recording_file = path
        self._state.recording_complete = True

    def draw(self):
        from imgui_bundle import imgui, implot

        # Reload if current_audio_file changed
        if self._state.current_audio_file != self._loaded_file and self._state.current_audio_file:
            self._load_audio(self._state.current_audio_file)

        # Update play position
        if self._playing:
            elapsed = time.time() - self._play_start_time
            self._play_pos = min(elapsed / self._duration, 1.0) if self._duration > 0 else 0.0
            if self._play_pos >= 1.0:
                self._playing = False
                self._play_pos = 0.0

        # ── Waveform ──────────────────────────────────────────────────────────
        if self._samples is not None:
            implot.begin_plot("##waveform", size=(-1, 80))
            implot.plot_line("##wave", self._samples)
            # Playhead
            if self._duration > 0:
                pos = self._play_pos * len(self._samples)
                implot.plot_vertical_lines("##playhead", np.array([pos], dtype=np.float64))
            implot.end_plot()
        else:
            imgui.text("No audio loaded.")

        # ── File info ─────────────────────────────────────────────────────────
        if self._loaded_file:
            import os
            fname = os.path.basename(self._loaded_file)
            imgui.text(f"{fname}  |  {self._sample_rate} Hz  |  {self._duration:.1f}s")

        # ── Transport controls ────────────────────────────────────────────────
        if imgui.button("Play" if not self._playing else "Pause"):
            if not self._playing:
                self._play()
            else:
                self._stop()
        imgui.same_line()
        if imgui.button("Stop"):
            self._stop()

        # Seek slider
        if self._duration > 0:
            imgui.same_line()
            imgui.set_next_item_width(150)
            changed, val = imgui.slider_float("##seek", self._play_pos, 0.0, 1.0, format="%.2f")
            if changed:
                self._play_pos = val

        # ── Recorder (RVC mode only) ──────────────────────────────────────────
        from src.enums.engine_type import EngineType
        if self._state.engine_type == EngineType.RVC:
            imgui.separator()
            imgui.text("Recorder:")
            imgui.same_line()
            if not self._recording:
                if imgui.button("Record"):
                    self._start_recording()
            else:
                elapsed = time.time() - self._rec_start_time
                imgui.text_colored((1.0, 0.3, 0.3, 1.0), f"Recording {elapsed:.1f}s")
                imgui.same_line()
                if imgui.button("Stop Recording"):
                    self._stop_recording()

        # ── Device selection ─────────────────────────────────────────────────
        if imgui.collapsing_header("Audio Devices"):
            imgui.text("Output:")
            imgui.same_line()
            imgui.set_next_item_width(200)
            changed, self._out_device_idx = imgui.combo(
                "##out_dev", self._out_device_idx, self._output_devices
            )
            imgui.text("Input:")
            imgui.same_line()
            imgui.set_next_item_width(200)
            changed, self._in_device_idx = imgui.combo(
                "##in_dev", self._in_device_idx, self._input_devices
            )
```

- [ ] **Step 2: Verify import**

```bash
python -c "from src.ui_imgui.widgets.audio_panel import AudioPanel; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/ui_imgui/widgets/audio_panel.py
git commit -m "feat: add AudioPanel with waveform (implot), transport, recorder"
```

---

## Phase 3: Integration

> Run after ALL Phase 2 streams are complete.

---

### Task 19: Migrate utils/ from QMetaObject to AppCallbacks

**Spec:** `docs/superpowers/specs/2026-03-18-imgui-migration-design.md` §10

**Files:**
- Modify: `src/utils/model_utils.py`
- Modify: `src/utils/inference_utils.py`
- Modify: `src/utils/bulk_utils.py`
- Modify: `src/utils/huggingface_utils.py`
- Modify: `src/tts_engines/upscale_engine.py`

The `parent` argument to all functions in these files is currently typed as `FallTalkApp` (a Qt widget). After migration it will be an `AppCallbacks` instance.

- [ ] **Step 1: Find all QMetaObject.invokeMethod call sites**

```bash
grep -rn "QMetaObject.invokeMethod" src/utils/ src/tts_engines/
```

- [ ] **Step 2: Replace each call site using this mapping**

| Old QMetaObject call | New AppCallbacks call |
|---|---|
| `invokeMethod(parent, "afterModelLoader", ...)` | `parent.on_model_loaded()` |
| `invokeMethod(parent, "after_engine_load", ..., engine)` | `parent.on_model_loaded()` |
| `invokeMethod(parent, "continueLoad", ...)` | `parent.on_continue_load()` |
| `invokeMethod(parent, "onError", ..., title, text)` | `parent.on_error(title, text)` |
| `invokeMethod(parent, "onWarn", ..., title, text)` | `parent.on_warn(title, text)` |
| `invokeMethod(parent, "update_loader", ..., msg)` | `parent.on_progress(msg)` |
| `invokeMethod(parent, "updateMediaplayer", ..., path)` | `parent.on_media_update(path)` |
| `invokeMethod(parent, "afterGen", ...)` | `parent.on_done()` |
| `invokeMethod(parent, "afterDownload", ...)` | `parent.on_done()` |
| `invokeMethod(parent, "afterModelDownload", ...)` | `parent.on_done()` |
| `invokeMethod(parent, "close_loader", ...)` | `parent.on_done()` |
| `invokeMethod(parent, "after_upscale", ...)` | `parent.on_done()` |

- [ ] **Step 3: Remove PySide6 imports from utils files**

```bash
grep -n "from PySide6" src/utils/model_utils.py src/utils/inference_utils.py src/utils/bulk_utils.py src/utils/huggingface_utils.py
```

Remove lines like:
```python
import PySide6
from PySide6.QtCore import QMetaObject, Qt, Q_ARG
```

- [ ] **Step 4: Update TYPE_CHECKING imports**

Change `from src.FallTalk import FallTalkApp` to `from src.ui_imgui.state import AppCallbacks` in each file's `TYPE_CHECKING` block. Update function signatures:

```python
# Before
def load_model(parent: 'FallTalkApp', character=None, ...):

# After
def load_model(parent: 'AppCallbacks', character=None, ...):
```

- [ ] **Step 5: Fix parent attribute accesses**

`model_utils.py` accesses `parent.tts_engine`, `parent.models`, `parent.characters_data` etc. directly. These need to be passed as arguments instead, or accessed via the `AppCallbacks` holding a reference to `AppState`.

Update `AppCallbacks` in `state.py` to carry a `state` reference for read-only access:

```python
# Add to AppCallbacks
state_ref: Optional[AppState] = None  # read-only access for utils that need model data
```

Then in `model_utils.py`, replace `parent.tts_engine` with `parent.state_ref.tts_engine`, etc.

- [ ] **Step 6: Apply same replacement to src/tts_engines/upscale_engine.py**

`upscale_engine.py` imports `QMetaObject` and calls it in its upscaling methods. Apply identical replacement:
- Remove `import PySide6`, `from PySide6.QtCore import QMetaObject, Qt, Q_ARG`
- Replace all `QMetaObject.invokeMethod(parent, ...)` with `parent.<callback>()` using the same mapping table above
- The `parent` parameter in `upscale_engine.py` methods becomes `AppCallbacks`

- [ ] **Step 7: Run a quick smoke test**

```bash
python -c "from src.utils.model_utils import load_model, get_character_model; print('ok')"
python -c "from src.utils.inference_utils import generic_inference, preprocess_text; print('ok')"
python -c "from src.utils.bulk_utils import bulk_inference; print('ok')"
python -c "from src.tts_engines.upscale_engine import UpscaleEngine; print('ok')"
```
Expected: All print `ok` (no ImportError)

- [ ] **Step 8: Commit**

```bash
git add src/utils/model_utils.py src/utils/inference_utils.py src/utils/bulk_utils.py src/utils/huggingface_utils.py src/tts_engines/upscale_engine.py
git commit -m "feat: replace QMetaObject.invokeMethod with AppCallbacks in utils and tts_engines"
```

---

### Task 20: Migrate falltalkapi.py

**Files:**
- Modify: `src/api/falltalkapi.py`

- [ ] **Step 1: Find QMetaObject calls**

```bash
grep -n "QMetaObject" src/api/falltalkapi.py
```

- [ ] **Step 2: Replace with api_command_queue**

```python
# Before
QMetaObject.invokeMethod(self.falltak_app, "engine_change", Qt.QueuedConnection, Q_ARG(str, engine))

# After
self.falltak_app.api_command_queue.put({"action": "engine_change", "engine": engine})
```

The `falltak_app` attribute should now hold an `AppState` instance (passed in at construction) instead of the Qt app widget.

- [ ] **Step 3: Remove PySide6 imports**

- [ ] **Step 4: Verify import**

```bash
python -c "from src.api.falltalkapi import FallTalkAPI; print('ok')"
```

- [ ] **Step 5: Commit**

```bash
git add src/api/falltalkapi.py
git commit -m "feat: replace QMetaObject in falltalkapi.py with api_command_queue"
```

---

### Task 21: Entry point swap

**Files:**
- Modify: `FallTalk.py`

- [ ] **Step 1: Back up current entry point**

```bash
cp FallTalk.py FallTalk_qt_backup.py
git add FallTalk_qt_backup.py
git commit -m "chore: backup Qt entry point before switchover"
```

- [ ] **Step 2: Replace FallTalk.py**

```python
#!/usr/bin/env python3
"""FallTalk — imgui_bundle entry point."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ui_imgui.app import run

if __name__ == "__main__":
    run()
```

- [ ] **Step 3: Test launch**

```bash
python FallTalk.py
```
Expected: hello_imgui window opens with FallTalk activity bar and workspaces.

- [ ] **Step 4: Commit**

```bash
git add FallTalk.py
git commit -m "feat: switch entry point to imgui_bundle UI"
```

---

### Task 22: Delete old UI code

- [ ] **Step 1: Verify all old code is unused**

```bash
grep -rn "from src.widgets" src/ui_imgui/ src/utils/ src/api/ FallTalk.py
grep -rn "from src.settings" src/ui_imgui/ src/utils/ src/api/ FallTalk.py
grep -rn "from src.help" src/ui_imgui/ src/utils/ src/api/ FallTalk.py
```
Expected: no matches

- [ ] **Step 2: Delete old directories**

```bash
git rm -r src/widgets/ src/settings/ src/help/ src/ui/
```

- [ ] **Step 3: Remove qfluentwidgets and PySide6 from requirements**

Edit `requirements.txt` (or `pyproject.toml`): remove `PySide6`, `qfluentwidgets`, `packaging` (if no longer needed). Add `imgui-bundle`, `sounddevice`, `soundfile`.

- [ ] **Step 4: Final smoke test**

```bash
pytest tests/ -v
python -c "from src.config.config import cfg; print(cfg.get(cfg.engine))"
python -c "from src.ui_imgui.app import run; print('entry point ok')"
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: delete old Qt widgets, settings, help files after imgui migration"
```

---

## Post-Migration Checklist

- [ ] All 16 engine types load and generate audio
- [ ] Reference audio selection works (BSA/XWM browser)
- [ ] Bulk CSV generation works end-to-end
- [ ] RVC mic recording + inference works
- [ ] Chat workspace (Qwen LLM) works
- [ ] Upscale workspace works
- [ ] Settings persist across restarts
- [ ] Old config (configv2.json) migrates correctly on first run
- [ ] API-only mode starts headlessly
- [ ] Delete `FallTalk_qt_backup.py` once stable
