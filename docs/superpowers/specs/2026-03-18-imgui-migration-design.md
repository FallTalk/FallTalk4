# FallTalk: PySide6/FluentWidgets → imgui_bundle Migration

**Date:** 2026-03-18
**Status:** Approved
**Scope:** Big-bang rewrite of all UI code. Feature parity with current app.

---

## 1. Goals & Constraints

- Remove all PySide6 and qfluentwidgets dependencies
- Adopt imgui_bundle (hello_imgui + imgui + implot) as the sole UI framework
- Maintain full feature parity with the current application
- Keep `cfg.get()` / `cfg.set()` config API unchanged across all consumer files
- `src/utils/`, `src/tts_engines/`, `src/api/` require targeted changes only (remove `QMetaObject` callbacks — see §10)
- Reference implementation: `I:\Fallout4Mods\Fallout4MCP\ui`

---

## 2. Migration Strategy

**Big-bang rewrite** into a new `src/ui_imgui/` directory alongside the existing `src/widgets/`. The old UI remains intact and runnable until the new UI reaches feature parity. Entry point (`FallTalk.py`) is switched once all workspaces are complete. Old `src/widgets/`, `src/settings/`, `src/help/`, `src/ui/` directories are deleted after switchover.

---

## 3. Top-Level Architecture

Modeled directly on the reference implementation's workspace protocol.

```
src/ui_imgui/
├── app.py                  # FallTalkApp — hello_imgui bootstrap, RunnerParams, Darcula theme
├── workspace.py            # Workspace protocol (get_dockable_windows, draw, draw_menu, etc.)
├── activity_bar.py         # Left icon sidebar for workspace switching
├── shared_panels.py        # AudioPanel (persistent across all workspaces)
├── state.py                # AppState — all shared mutable application state + AppCallbacks
├── workspaces/
│   ├── characters.py       # Characters + References panels
│   ├── generation.py       # Generation + Multi-gen + RVC + Settings + Help panels
│   ├── bulk.py             # BulkCSV + BulkRVC + BulkFUZ + EzVoice panels
│   ├── chat.py             # Chat panel
│   ├── upscale.py          # Upscale panel
│   └── settings.py         # Global settings + FAQ panels
└── widgets/
    ├── engine_settings.py  # Replaces all 20 src/settings/*.py files
    ├── engine_help.py      # Replaces all 20+ src/help/*.py files
    ├── drawer.py           # Slide-in overlay panel
    ├── audio_panel.py      # Dockable audio player + waveform (implot) + recorder
    ├── tables.py           # Reusable imgui table helpers
    ├── file_dialog.py      # tkinter-backed file/folder pickers
    └── common.py           # Loader overlay, error flyouts, shared helpers
```

---

## 4. AppState (`state.py`)

A plain Python dataclass/object holding all shared mutable state. Passed by reference to every workspace and panel. Replaces the tangled `self.*` attributes on the current `FallTalkApp`.

Also defines `AppCallbacks` — a plain dataclass replacing all `QMetaObject.invokeMethod` calls in `src/utils/` (see §10).

**Key fields:**
```python
@dataclass
class AppCallbacks:
    """Replaces QMetaObject.invokeMethod cross-thread dispatch in src/utils/."""
    on_progress: Callable[[str], None]        # update loader message
    on_done: Callable[[], None]               # complete loader, re-enable UI
    on_error: Callable[[str, str], None]      # title, message
    on_warn: Callable[[str, str], None]       # title, message
    on_model_loaded: Callable[[], None]       # after model load complete
    on_media_update: Callable[[str], None]    # new audio file path
    on_continue_load: Callable[[], None]      # after engine change, trigger pending op

class AppState:
    # Engine
    tts_engine: Optional[tts_engine] = None
    engine_type: EngineType = EngineType(cfg.get(cfg.engine))

    # Models / characters
    characters_data: dict = {}
    models: dict = {}
    shared_models: Optional[list] = None
    custom_models: Optional[dict] = None
    default_references: Optional[dict] = None

    # Reference audio
    reference_audio: list = []
    reference_audio_length: float = 0.0

    # Pending loads
    pending_character: Optional[str] = None
    pending_model: Optional[dict] = None
    pending_rvc: Optional[str] = None
    pending_base: bool = False
    pending_bulk: bool = False
    pending_ez: bool = False

    # Audio
    current_audio_file: Optional[str] = None
    recording_file: Optional[str] = None      # set by recorder thread when recording stops
    recording_complete: bool = False           # polled by GenerationWorkspace each frame

    # Loader
    loading: bool = False
    loading_message: str = ""

    # Errors (polled each frame, cleared after display)
    error_queue: queue.Queue = field(default_factory=queue.Queue)

    # Engines
    upscale_engine = None
    apbwe_engine = None
    transcription_engine = None

    # Startup flags
    first_start: bool = False
    disclaimer_accepted: bool = False
    api_only_mode: bool = False

    # Callbacks (built at app init, passed to utils functions as parent-replacement)
    callbacks: Optional[AppCallbacks] = None

    # Engine change subscribers (replaces Qt OnEngineChange signal)
    on_engine_change: list[Callable] = field(default_factory=list)

    # API cross-thread queue (replaces QMetaObject in falltalkapi.py)
    api_command_queue: queue.Queue = field(default_factory=queue.Queue)
```

---

## 5. Workspace / Panel Mapping

### CharactersWorkspace
- **Characters Panel** — table of trained/untrained/custom models with search, load/download/delete/update actions. Tabs: Trained | Untrained | Custom.
- **References Panel** — reference audio table with BSA/XWM browser, search, duration tracker, audio preview trigger.

### GenerationWorkspace
- **Generation Panel** — text input, generate button, output naming, RVC toggle, upscaler toggle. Engine-specific controls (F5 word-level edit dropdowns) rendered inline.
- **Engine Settings Panel** — dockable; engine-aware settings rendered by `draw_engine_settings(state)`.
- **Engine Help Panel** — dockable; rendered by `draw_engine_help(state)`.
- **Multi-Generation Panel** — results cards with per-audio save/play/select radio buttons.
- Each frame: check `state.recording_complete`; if True, clear it and dispatch RVC inference.

### BulkWorkspace
- **Bulk CSV Panel** — table editor, CSV import, generate button.
- **Bulk RVC Panel** — folder picker, character selector, thread count.
- **Bulk FUZ Panel** — folder picker, include-subdirs toggle.
- **EzVoice Panel** — ESP dialogue table editor, voice/reference selection, bulk generate.

### ChatWorkspace
- **Chat Panel** — dialogue history, text input, LLM-backed (Qwen) response generation, audio playback per message.

### UpscaleWorkspace
- **Upscale Panel** — mode (Denoise/Isolate/Upscale), folder picker, sample rate, include-subdirs, replace-existing.

### SettingsWorkspace
- **Main Settings Panel** — all global config organized by category (General, Device, Paths, Text Processing, Audio, Bulk, API). Includes a "Reset to Defaults" button that calls `cfg.reset()` (see §6). Qt-only keys (`themeColor`, `dpiScale`, `enableAcrylicBackground`, `minimizeToTray`, `language`) are dropped; DPI scale is replaced with imgui `FontGlobalScale` adjustment.
- **FAQ Panel** — `imgui.collapsing_header` sections replacing the FluentWidgets accordion.

### Persistent (all workspaces)
- **Audio Panel** — waveform display (implot), play/pause/stop/seek, recording controls for RVC mic mode, file info bar.

---

## 6. Config System Migration

**File:** `src/config/config.py` — internals replaced, public API unchanged.

### What changes
- Remove: `QConfig`, `ConfigItem`, `RangeConfigItem`, `OptionsConfigItem`, all qfluentwidgets imports
- Remove: `ConfigValidator`, `OptionsValidator`, `RangeValidator`, `EnumSerializer`, `FolderValidator`, `LanguageSerializer`
- Remove: `OnEngineChange` Qt signal
- Remove Qt-only keys: `themeColor`, `dpiScale`, `enableAcrylicBackground`, `minimizeToTray`, `language`
- Replace with: plain `Config` class backed by `config/settings.json`

### New internal structure
```python
class ConfigKey:
    def __init__(self, key: str, default, type_=None):
        self.key = key
        self.default = default
        self.type_ = type_

class Config:
    engine = ConfigKey("engine", "XTTSv2")
    device = ConfigKey("device", "cuda")
    speed = ConfigKey("speed", 1.0, float)
    # ... all existing non-Qt keys ...

    def __init__(self):
        self._data = {}
        self._defaults = {}  # populated from ConfigKey defaults at init
        self._path = os.path.join(get_app_root(), "config", "settings.json")
        self.load()

    def get(self, key: ConfigKey):
        val = self._data.get(key.key, key.default)
        return key.type_(val) if key.type_ and val is not None else val

    def set(self, key: ConfigKey, value):
        self._data[key.key] = value
        self.save()

    def reset(self):
        """Reset all settings to defaults and save."""
        self._data = dict(self._defaults)
        self.save()

    def load(self): ...   # See migration path below
    def save(self): ...   # Atomic JSON write via temp file + rename
```

### Migration compatibility — old config format
The old qfluentwidgets config file is `config/configv2.json` stored in **QSettings INI format** (not plain JSON). On first load:
1. Check if `configv2.json` exists and `settings.json` does not
2. If so, parse `configv2.json` using Python's `configparser` (it is an INI file with a `[General]` section)
3. Map each key-value, applying type coercion:
   - Enum-valued keys (e.g., `rvc_pitch_extraction = "rmvpe"`) → store string as-is; `cfg.get()` returns the enum via `type_(val)` coercion
   - Bool values stored as `"true"`/`"false"` strings → coerce to `bool`
   - Numeric strings → coerce via `float()` or `int()`
4. Write migrated values to `settings.json`
5. Do not delete `configv2.json` (non-destructive)

If neither file exists, `settings.json` is created fresh with all defaults.

Unknown keys in JSON silently ignored. All 50+ consumer files: **zero changes required**.

---

## 7. Audio Panel

**File:** `src/ui_imgui/widgets/audio_panel.py`

- Persistent dockable panel visible in all workspaces
- Observes `state.current_audio_file` each frame; reloads waveform on change (checks against `_loaded_file` cache)
- **Waveform:** `implot.plot_line()` of loaded audio samples (downsampled to ~2000 points via `soundfile.read()` + numpy stride). Colored region overlay shows selected reference range. Click-to-seek via `implot.is_plot_hovered()` + mouse position.
- **Transport:** Play/Pause/Stop buttons + current time / total duration. Playback via `sounddevice.play()` in an `AsyncWorker` thread. Current position polled via `sounddevice.get_stream_pos()` if available, else time-delta estimate.
- **Device selection:** `sounddevice.query_devices()` for enumeration. Device names displayed in `imgui.combo`. Selected device index stored in `cfg` (new key `audio_output_device` / `audio_input_device`). Passed to `sounddevice.play(device=idx)` / `sounddevice.InputStream(device=idx)`.
- **Recorder:** Shown only when `state.engine_type == EngineType.RVC`. Record/Stop button, duration counter. Uses `sounddevice.InputStream` at 40000 Hz, writes WAV via `soundfile`. On stop: sets `state.recording_file = path`, sets `state.recording_complete = True`.
- **File info bar:** filename, sample rate, duration.

---

## 8. Drawer Overlay

**File:** `src/ui_imgui/widgets/drawer.py`

Replaces the animated `RightDrawer`. Implementation:
- Fixed-position `imgui.begin()` window pinned to right edge of main viewport
- `_open_pct: float` (0.0→1.0) lerped each frame using delta time — smooth slide without Qt animation system
- `_content_fn: Callable | None` — passed in on open, called inside the drawer each frame
- Close button + click-outside detection via `imgui.is_mouse_clicked()` + bounds check

Usage in GenerationWorkspace:
```python
# Settings button
if imgui.button("Settings"):
    self._drawer.open(lambda: draw_engine_settings(self._state))

# Help button
if imgui.button("Help"):
    self._drawer.open(lambda: draw_engine_help(self._state))

# Each frame
self._drawer.draw()
```

---

## 9. Settings & Help Consolidation

**40+ files → 2 files.**

### `widgets/engine_settings.py`
Single `draw_engine_settings(state: AppState)` function. `match state.engine_type` dispatches to per-engine imgui rendering. Each case uses `imgui.slider_float`, `imgui.slider_int`, `imgui.checkbox`, `imgui.combo` reading from and writing to `cfg` directly. No classes, no inheritance.

### `widgets/engine_help.py`
Single `draw_engine_help(state: AppState)` function. `match state.engine_type` renders help content as `imgui.text_wrapped()` + `imgui.collapsing_header()` sections. All help text as Python string constants at top of file.

**Deleted after migration:** `src/settings/` (20 files), `src/help/` (20+ files).

---

## 10. Threading, Callbacks & Loader Overlay

### The QMetaObject Problem

`src/utils/model_utils.py`, `src/utils/inference_utils.py`, `src/utils/bulk_utils.py`, and `src/utils/huggingface_utils.py` all use `QMetaObject.invokeMethod(parent, "slotName", ...)` for thread-safe cross-thread callbacks into the UI. These **must be replaced** — they are the only Qt coupling in `src/utils/`.

### Replacement: AppCallbacks

All utils functions accept a `parent` argument (currently the Qt app). After migration, `parent` is replaced with an `AppCallbacks` instance (defined in `state.py`, built at app init). All `QMetaObject.invokeMethod` call sites in utils become direct callable invocations:

```python
# Before (Qt)
QMetaObject.invokeMethod(parent, "update_loader", Qt.QueuedConnection, Q_ARG(str, msg))
QMetaObject.invokeMethod(parent, "afterModelLoader", Qt.QueuedConnection, Q_ARG(QObject, parent))
QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(QObject, parent), Q_ARG(str, title), Q_ARG(str, text))

# After (plain Python — safe from any thread via AppState fields)
parent.on_progress(msg)       # writes state.loading_message
parent.on_model_loaded()      # writes state.loading = False
parent.on_error(title, text)  # puts (title, text) on state.error_queue
```

`AppCallbacks` is constructed in `app.py` at startup and stored as `state.callbacks`. It is passed to all utils functions as the `parent` argument. The main loop drains `state.error_queue` each frame and shows error flyouts.

**Files requiring changes in `src/utils/`:**
- `model_utils.py` — replace all `QMetaObject.invokeMethod` with `parent.<callback>()`
- `inference_utils.py` — same
- `bulk_utils.py` — same
- `huggingface_utils.py` — same

These are **targeted surgical changes** — only the `QMetaObject` call sites change, all inference/model logic is untouched.

### falltalkapi.py

`src/api/falltalkapi.py` uses `QMetaObject.invokeMethod` to dispatch commands cross-thread into the Qt app. Replacement:

```python
# Before
QMetaObject.invokeMethod(self.falltak_app, "engine_change", Qt.QueuedConnection, Q_ARG(str, engine))

# After — post command to queue, drained by main loop each frame
self.falltak_app.api_command_queue.put({"action": "engine_change", "engine": engine})
```

`app.py`'s main loop drains `state.api_command_queue` each frame and dispatches commands. This is a Stream 7 task.

### AsyncWorker

All heavy operations run in daemon threads. Adopt `AsyncWorker` from reference:
```python
class AsyncWorker:
    def __init__(self, target, args=()):
        self._thread = threading.Thread(target=self._run, args=args, daemon=True)
        self.done = False
        self.error = None
    def start(self): self._thread.start()
```
Each workspace holds `_worker: AsyncWorker | None`, polls `worker.done` each frame.

### Loader Overlay (`widgets/common.py`)
Replaces `StateToolTip`. When `state.loading = True`, every workspace renders a centered semi-transparent overlay window with spinner and `state.loading_message`. Blocks interaction via no-move/no-resize/no-input flags.

`state.loading` and `state.loading_message` are written from background threads via `AppCallbacks.on_progress()` (safe — simple Python field assignment, read each frame by main thread).

### Error Flyouts
Drain `state.error_queue` each frame. Display as small `imgui.begin_popup` near the center of the screen. Auto-dismiss after 5 seconds via timestamp.

---

## 11. Startup Flow

`app.py` handles the startup sequence before the main imgui loop begins:

1. Load config (`cfg.load()`) — migrate old format if needed
2. Check `cfg.get(cfg.api_only_mode)`: if True, start FastAPI server but **do not call `hello_imgui.run()`** — run in headless mode
3. Otherwise, call `hello_imgui.run(runner_params)` with a `post_init` callback that:
   a. Applies Darcula theme
   b. Loads static JSON data (characters, models, shared models, default references)
   c. Checks `first_start` / `accepted_disclaimer` — opens a disclaimer modal on first frame via `imgui.open_popup`
   d. Checks `fallout_4_directory` — opens a warning modal if not found
   e. If `check_for_updates`: spawns `AsyncWorker` to fetch latest release
   f. If `load_engine_art_start`: triggers engine load for configured engine

**Disclaimer modal:** Uses deferred-open pattern (`imgui.open_popup` called on first frame, `imgui.begin_popup_modal` renders it). User must click "Agree" or app exits. On agree: `cfg.set(cfg.accepted_disclaimer, True)`.

---

## 12. Agent Team Execution Plan

Stream 1 must complete before others start. Streams 2–6 run in parallel. Stream 7 runs after all others complete.

> **Note:** Stream 6 (Audio Panel) depends on `state.recording_file` and `state.recording_complete` being defined in Stream 1's `state.py`. These fields must be finalized in Stream 1 before Stream 6 begins.

### Stream 1 — Foundation (blocks all others)
- `src/config/config.py` — remove Qt, keep API, add `reset()`, handle INI migration
- `src/ui_imgui/app.py` — hello_imgui bootstrap + Darcula theme + startup flow
- `src/ui_imgui/state.py` — AppState + AppCallbacks (fully specify all fields incl. recording)
- `src/ui_imgui/workspace.py` — protocol
- `src/ui_imgui/activity_bar.py`
- `src/ui_imgui/widgets/common.py` — loader overlay, error flyouts, queue draining
- `src/ui_imgui/widgets/drawer.py`
- `src/ui_imgui/widgets/file_dialog.py` — tkinter-backed pickers
- `src/ui_imgui/widgets/tables.py` — reusable table helpers

### Stream 2 — Characters & References Workspace
- `src/ui_imgui/workspaces/characters.py`
- Characters panel: table, search, load/download/delete/update actions, tabs
- References panel: audio table, BSA/XWM browser, search, duration tracker

### Stream 3 — Generation Workspace
- `src/ui_imgui/workspaces/generation.py` (incl. `recording_complete` polling)
- `src/ui_imgui/widgets/engine_settings.py` (replaces 20 settings files)
- `src/ui_imgui/widgets/engine_help.py` (replaces 20+ help files)
- Generation panel, Engine Settings panel, Engine Help panel, Multi-generation panel

### Stream 4 — Bulk & EzVoice Workspace
- `src/ui_imgui/workspaces/bulk.py`
- Bulk CSV, Bulk RVC, Bulk FUZ, EzVoice panels

### Stream 5 — Chat, Upscale, Settings Workspaces
- `src/ui_imgui/workspaces/chat.py`
- `src/ui_imgui/workspaces/upscale.py`
- `src/ui_imgui/workspaces/settings.py` (global settings + FAQ, FontGlobalScale DPI)

### Stream 6 — Audio Panel (depends on Stream 1 state.py recording fields)
- `src/ui_imgui/widgets/audio_panel.py`
- `src/ui_imgui/shared_panels.py`
- Waveform (implot), transport controls, sounddevice device enumeration, recorder

### Stream 7 — Entry Point Swap & Cleanup (after all streams)
- Update `FallTalk.py` to instantiate `FallTalkApp` from `src/ui_imgui/app.py`
- Update `src/utils/model_utils.py`, `inference_utils.py`, `bulk_utils.py`, `huggingface_utils.py` — replace all `QMetaObject.invokeMethod` with `AppCallbacks` calls
- Update `src/api/falltalkapi.py` — replace `QMetaObject.invokeMethod` with `api_command_queue`
- Delete `src/widgets/`, `src/settings/`, `src/help/`, `src/ui/cards.py`
- Final integration testing

---

## 13. Files Deleted After Migration

| Directory / File | Count | Reason |
|---|---|---|
| `src/widgets/*.py` | 21 files | Replaced by `src/ui_imgui/workspaces/` + `widgets/` |
| `src/settings/*.py` | 20 files | Consolidated into `engine_settings.py` |
| `src/help/*.py` | 20+ files | Consolidated into `engine_help.py` |
| `src/ui/cards.py` | 1 file | Qt custom cards, not needed |

**Total: ~62 files deleted, ~11,500 LOC removed.**

---

## 14. Dependencies

### Added
```
imgui-bundle>=1.5.0
implot  (bundled with imgui-bundle)
sounddevice
soundfile
```

### Removed
```
PySide6
qfluentwidgets
packaging  (only used for version comparison — replace with stdlib importlib.metadata or keep)
```
