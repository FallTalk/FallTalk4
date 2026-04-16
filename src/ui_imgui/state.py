from __future__ import annotations

import queue
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import logging

from src.config.config import cfg

if TYPE_CHECKING:
    from src.tts_engines import tts_engine
    from src.enums.engine_type import EngineType

logger = logging.getLogger('falltalk')


def _extract_loading_progress(message: str) -> Optional[float]:
    if not message:
        return None

    fraction_match = re.search(r'(\d+)\s*/\s*(\d+)', message)
    if fraction_match:
        current = int(fraction_match.group(1))
        total = int(fraction_match.group(2))
        if total > 0:
            return max(0.0, min(1.0, current / total))

    percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', message)
    if percent_match:
        percent = float(percent_match.group(1))
        return max(0.0, min(1.0, percent / 100.0))

    return None


@dataclass
class AppCallbacks:
    """
    Replaces QMetaObject.invokeMethod cross-thread UI dispatch in src/utils/.
    Pass an instance of this as the `parent` argument to all utils functions
    (load_model, generic_inference, bulk_inference, etc.).
    """
    on_progress: Callable[[str], None]      # update loader message
    on_done: Callable[[], None]             # complete loader, clear loading flag
    on_error: Callable[[str, str], None]    # (title, message) -> error queue
    on_warn: Callable[[str, str], None]     # (title, message) -> warn queue
    on_model_loaded: Callable[[], None]     # after model load -- enable widget
    on_media_update: Callable[[str], None]  # new output audio file path
    on_continue_load: Callable[[], None]    # after engine swap, resume pending op
    state_ref: Optional[AppState] = None   # read-only access for utils needing model/engine data

    @classmethod
    def from_state(cls, state: AppState) -> AppCallbacks:
        """Build a callbacks instance backed by the given AppState."""

        def _on_media_update(path: str):
            logger.debug(f"on_media_update: path={path}")
            state.current_audio_file = path
            if cfg.get(cfg.auto_play):
                logger.debug("on_media_update: auto_play enabled, requesting playback")
                state.play_audio_requested = True

        def _push_loading_message(msg: str):
            clean_msg = (msg or "").strip()
            now = time.time()
            if not state.loading:
                state.loading_started_at = now
                state.loading_history.clear()
            state.loading = True
            state.loading_message = clean_msg
            state.loading_progress = _extract_loading_progress(clean_msg)
            if clean_msg and (not state.loading_history or state.loading_history[-1] != clean_msg):
                state.loading_history.append(clean_msg)
                state.loading_history[:] = state.loading_history[-4:]

        def _clear_loading():
            state.loading = False
            state.loading_message = ""
            state.loading_progress = None
            state.loading_started_at = 0.0
            state.loading_history.clear()

        return cls(
            on_progress=_push_loading_message,
            on_done=_clear_loading,
            on_error=lambda title, msg: (
                _clear_loading(),
                state.error_queue.put((title, msg)),
            ),
            on_warn=lambda title, msg: state.warn_queue.put((title, msg)),
            on_model_loaded=lambda: setattr(state, 'model_just_loaded', True),
            on_media_update=_on_media_update,
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
    selected_references: set = field(default_factory=set)  # indices into reference_audio
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
    play_audio_requested: bool = False
    audio_volume: float = 1.0
    audio_panel_expanded: bool = False
    recording_file: Optional[str] = None
    recording_complete: bool = False    # set True by recorder thread; cleared by GenerationWorkspace after consuming

    # Loader
    loading: bool = False
    loading_message: str = ""
    loading_progress: Optional[float] = None
    loading_started_at: float = 0.0
    loading_history: list[str] = field(default_factory=list)

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

    # Shared generation state (used by Generation and Multi-Generation pages)
    text_input: str = ""
    output_name: str = ""
    nav_collapsed: bool = field(default_factory=lambda: cfg.get(cfg.nav_collapsed))

    # Startup modal flags (polled by app.py show_gui on first frames)
    _show_disclaimer: bool = False
    _show_fallout_warning: bool = False
