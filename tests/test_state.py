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
