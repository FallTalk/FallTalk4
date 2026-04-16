from __future__ import annotations

import os
import time
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState


class RecorderPanel:
    """Audio recorder panel (used for RVC voice input)."""

    HEIGHT = 96

    def __init__(self, state: AppState):
        self._state = state
        self._recording: bool = False
        self._rec_start_time: float = 0.0
        self._rec_stream = None
        self._rec_frames: list = []
        self._input_devices: list[str] = []
        self._in_device_idx: int = 0
        self._enumerate_devices()

    def _enumerate_devices(self):
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            self._input_devices = [d['name'] for d in devices if d['max_input_channels'] > 0]
        except Exception:
            self._input_devices = ["Default"]

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
        from src.utils.file_utils import formatted_time_stamp_uuid
        path = os.path.join("temp", f"recording_{formatted_time_stamp_uuid()}.wav")
        os.makedirs("temp", exist_ok=True)
        data = np.concatenate(self._rec_frames, axis=0)
        sf.write(path, data, 40000)
        self._state.recording_file = path
        self._state.recording_complete = True

    @property
    def should_show(self) -> bool:
        from src.enums.engine_type import EngineType
        return self._state.engine_type == EngineType.RVC

    def _status_label(self) -> str:
        if self._recording:
            elapsed = time.time() - self._rec_start_time
            return f"Recording {elapsed:.1f}s"
        return "Idle"

    def draw(self):
        from imgui_bundle import imgui, icons_fontawesome_6 as fa

        imgui.text(f"{fa.ICON_FA_MICROPHONE}  Recorder")
        imgui.text_disabled("Status:")
        imgui.same_line()
        if self._recording:
            elapsed = time.time() - self._rec_start_time
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.3, 0.3, 1.0),
                f"{fa.ICON_FA_CIRCLE}  Recording {elapsed:.1f}s",
            )
        else:
            imgui.text(self._status_label())

        imgui.spacing()

        if not self._recording:
            if imgui.button(f"{fa.ICON_FA_CIRCLE}  Record"):
                self._start_recording()
        else:
            if imgui.button(f"{fa.ICON_FA_STOP}  Stop"):
                self._stop_recording()

        imgui.spacing()
        imgui.text_disabled("Input Device:")
        imgui.same_line()
        if self._input_devices:
            imgui.set_next_item_width(max(220, imgui.get_content_region_avail().x))
            changed, self._in_device_idx = imgui.combo(
                "##in_dev", self._in_device_idx, self._input_devices
            )
        else:
            imgui.text_disabled("No input devices found")
