from __future__ import annotations

import logging
import os
import time
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState

logger = logging.getLogger("falltalk.audio_panel")


def _format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(max(0, int(seconds)), 60)
    return f"{m:02d}:{s:02d}"


def _truncate_middle(text: str, max_len: int) -> str:
    """Truncate long text while preserving the start and end."""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    front = (max_len - 3) // 2
    back = max_len - 3 - front
    return f"{text[:front]}...{text[-back:]}"


class AudioPanel:
    """
    Audio player with compact and expanded modes.

    Compact: two-row transport view with explicit file, time, seek, and volume controls.
    Expanded: adds interactive implot waveform with draggable playhead and device selection.
    """

    COMPACT_HEIGHT = 92
    EXPANDED_HEIGHT = 290

    def __init__(self, state: AppState):
        self._state = state
        self._loaded_file: Optional[str] = None
        self._raw_data: Optional[np.ndarray] = None  # full audio for playback
        self._samples: Optional[np.ndarray] = None    # downsampled for waveform
        self._sample_rate: int = 44100
        self._duration: float = 0.0
        self._play_pos: float = 0.0  # 0.0 - 1.0
        self._playing: bool = False
        self._play_start_time: float = 0.0
        self._play_start_offset: float = 0.0  # seconds offset when play started
        self._output_devices: list[str] = []
        self._out_device_idx: int = 0
        self._enumerate_devices()

    def _enumerate_devices(self):
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            self._output_devices = []
            self._output_device_ids = []  # actual sounddevice indices
            for i, d in enumerate(devices):
                if d['max_output_channels'] > 0:
                    self._output_devices.append(d['name'])
                    self._output_device_ids.append(i)
            logger.info(f"Found {len(self._output_devices)} output devices: "
                        f"{list(zip(self._output_device_ids, self._output_devices))}")
            default_out = sd.query_devices(kind='output')
            logger.info(f"Default output device: {default_out['name']}")
        except Exception as e:
            logger.exception(f"Failed to enumerate audio devices: {e}")
            self._output_devices = ["Default"]
            self._output_device_ids = [None]

    def _load_audio(self, path: str):
        logger.info(f"_load_audio called: path={path!r}, exists={os.path.exists(path)}")
        try:
            import soundfile as sf
            data, sr = sf.read(path, always_2d=False)
            if data.ndim > 1:
                data = data[:, 0]  # mono
            self._raw_data = data.astype(np.float32)
            # Downsample to ~2000 points for waveform display
            step = max(1, len(data) // 2000)
            self._samples = data[::step].astype(np.float32)
            self._sample_rate = sr
            self._duration = len(data) / sr
            self._loaded_file = path
            self._play_pos = 0.0
            logger.info(f"_load_audio OK: sr={sr}, samples={len(data)}, duration={self._duration:.2f}s")
        except Exception as e:
            logger.exception(f"_load_audio FAILED: {e}")
            self._raw_data = None
            self._samples = None
            self._duration = 0.0

    def _play(self, from_pos: float = -1.0):
        """Start playback. If from_pos >= 0, seek to that normalized position first."""
        logger.info(f"_play called: from_pos={from_pos}, loaded_file={self._loaded_file!r}, "
                     f"raw_data={'None' if self._raw_data is None else f'shape={self._raw_data.shape}'}")
        if self._loaded_file is None or self._raw_data is None:
            logger.warning("_play aborted: no file or data loaded")
            return
        try:
            import sounddevice as sd
            sd.stop()

            if from_pos >= 0:
                self._play_pos = max(0.0, min(from_pos, 1.0))

            start_sample = int(self._play_pos * len(self._raw_data))
            remaining = self._raw_data[start_sample:]
            if len(remaining) == 0:
                logger.warning("_play aborted: no remaining samples")
                return

            # Apply volume
            vol = self._state.audio_volume
            play_data = remaining * vol if vol < 1.0 else remaining

            # Map combo index to actual sounddevice device id
            if self._out_device_idx < len(self._output_device_ids):
                sd_device = self._output_device_ids[self._out_device_idx]
            else:
                sd_device = None  # system default

            logger.info(f"sd.play: samples={len(play_data)}, sr={self._sample_rate}, "
                        f"device={sd_device}, volume={vol:.2f}, dtype={play_data.dtype}")
            self._playing = True
            self._play_start_offset = self._play_pos * self._duration
            self._play_start_time = time.time()
            sd.play(play_data, self._sample_rate, device=sd_device, blocksize=2048)
            logger.info(f"sd.play returned OK, sd.get_stream()={sd.get_stream()}")
        except Exception as e:
            logger.exception(f"_play FAILED: {e}")
            self._playing = False

    def _pause(self):
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self._playing = False

    def _stop(self):
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self._playing = False
        self._play_pos = 0.0

    def _seek(self, pos: float):
        """Seek to normalized position (0-1). If playing, restart from new position."""
        self._play_pos = max(0.0, min(pos, 1.0))
        if self._playing:
            self._play(from_pos=self._play_pos)

    def _display_file_name(self, max_len: int = 42) -> str:
        if not self._loaded_file:
            return "No audio loaded"
        return _truncate_middle(os.path.basename(self._loaded_file), max_len)

    def _time_label(self) -> str:
        current_sec = self._play_pos * self._duration
        return f"{_format_time(current_sec)} / {_format_time(self._duration)}"

    @property
    def height(self) -> int:
        return self.EXPANDED_HEIGHT if self._state.audio_panel_expanded else self.COMPACT_HEIGHT

    def draw(self):
        from imgui_bundle import imgui, icons_fontawesome_6 as fa

        # -- Handle state changes --
        if self._state.current_audio_file != self._loaded_file and self._state.current_audio_file:
            logger.info(f"draw: audio file changed, loading: {self._state.current_audio_file!r}")
            self._load_audio(self._state.current_audio_file)

        if self._state.play_audio_requested:
            logger.info(f"draw: play_audio_requested=True, file={self._state.current_audio_file!r}")
            self._state.play_audio_requested = False
            if self._state.current_audio_file:
                if self._loaded_file != self._state.current_audio_file:
                    self._load_audio(self._state.current_audio_file)
                self._play(from_pos=0.0)
            else:
                logger.warning("draw: play requested but current_audio_file is None")

        # -- Update play position --
        if self._playing:
            elapsed = time.time() - self._play_start_time
            current_time = self._play_start_offset + elapsed
            self._play_pos = min(current_time / self._duration, 1.0) if self._duration > 0 else 0.0
            if self._play_pos >= 1.0:
                self._playing = False
                self._play_pos = 0.0

        # -- Draw based on mode --
        if self._state.audio_panel_expanded:
            self._draw_expanded()
        else:
            self._draw_compact()

    def _draw_compact(self):
        from imgui_bundle import imgui, icons_fontawesome_6 as fa

        # Row 1: transport + file context.
        play_label = "Pause" if self._playing else "Play"
        if imgui.button(f"{fa.ICON_FA_PAUSE if self._playing else fa.ICON_FA_PLAY}  {play_label}"):
            if self._playing:
                self._pause()
            else:
                self._play()
        imgui.same_line()

        if imgui.button(f"{fa.ICON_FA_STOP}  Stop"):
            self._stop()
        imgui.same_line()

        imgui.text_disabled("File:")
        imgui.same_line()
        if self._loaded_file:
            imgui.text(self._display_file_name())
        else:
            imgui.text_disabled("No audio loaded")
        if self._loaded_file:
            imgui.same_line()
            imgui.text_disabled(f"{self._sample_rate} Hz")

        imgui.spacing()

        # Row 2: time + seek.
        imgui.text_disabled("Time:")
        imgui.same_line()
        imgui.text(self._time_label())
        imgui.same_line()

        seek_width = max(140, imgui.get_content_region_avail().x)
        imgui.set_next_item_width(seek_width)
        changed, val = imgui.slider_float(
            "##seek_compact", self._play_pos, 0.0, 1.0,
            format="",
        )
        if changed:
            self._seek(val)

        imgui.spacing()

        # Row 3: volume and expand/collapse.
        imgui.text_disabled("Volume:")
        imgui.same_line()
        vol = self._state.audio_volume
        if vol <= 0.01:
            vol_icon = fa.ICON_FA_VOLUME_XMARK
        elif vol < 0.5:
            vol_icon = fa.ICON_FA_VOLUME_LOW
        else:
            vol_icon = fa.ICON_FA_VOLUME_HIGH
        imgui.text(vol_icon)
        imgui.same_line()
        imgui.set_next_item_width(max(120, imgui.get_content_region_avail().x - 80))
        vol_pct = vol * 100.0
        changed, new_pct = imgui.slider_float("##vol", vol_pct, 0.0, 100.0, format="%.0f%%")
        if changed:
            self._state.audio_volume = new_pct / 100.0
        imgui.same_line()

        if imgui.button(f"{fa.ICON_FA_CHEVRON_UP}  Expand"):
            self._state.audio_panel_expanded = True

    def _draw_expanded(self):
        from imgui_bundle import imgui, implot, icons_fontawesome_6 as fa

        # -- Waveform with interactive playhead --
        waveform_height = 100
        if self._samples is not None:
            implot.push_style_var(implot.StyleVar_.plot_padding.value, imgui.ImVec2(0, 0))
            if implot.begin_plot("##waveform", size=imgui.ImVec2(-1, waveform_height),
                                 flags=implot.Flags_.no_legend.value
                                 | implot.Flags_.no_mouse_text.value
                                 | implot.Flags_.no_title.value):
                implot.setup_axes("", "",
                                  implot.AxisFlags_.no_decorations.value,
                                  implot.AxisFlags_.no_decorations.value)
                implot.setup_axes_limits(0, len(self._samples), -1.0, 1.0, implot.Cond_.always.value)

                # Shaded waveform
                implot.plot_shaded("##wave", self._samples, 0.0)

                # Draggable playhead
                if self._duration > 0:
                    playhead_x = self._play_pos * len(self._samples)
                    color = imgui.ImVec4(1.0, 0.9, 0.2, 1.0)  # yellow
                    changed, new_x, _, _, _ = implot.drag_line_x(
                        0, playhead_x, color, 2.0
                    )
                    if changed:
                        new_pos = max(0.0, min(new_x / len(self._samples), 1.0))
                        self._seek(new_pos)

                implot.end_plot()
            implot.pop_style_var()
        else:
            imgui.dummy(imgui.ImVec2(0, waveform_height))
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() - waveform_height / 2 - 8)
            imgui.text_disabled("    No audio loaded.")
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + waveform_height / 2 - 8)

        imgui.spacing()

        # -- File summary row --
        imgui.text_disabled("File:")
        imgui.same_line()
        if self._loaded_file:
            imgui.text(self._display_file_name(64))
        else:
            imgui.text_disabled("No audio loaded")
        if self._loaded_file:
            imgui.same_line()
            imgui.text_disabled(f"Sample rate: {self._sample_rate} Hz")
            imgui.same_line()
            imgui.text_disabled(f"Duration: {_format_time(self._duration)}")

        imgui.spacing()

        # -- Transport row --
        play_label = "Pause" if self._playing else "Play"
        if imgui.button(f"{fa.ICON_FA_PAUSE if self._playing else fa.ICON_FA_PLAY}  {play_label}##exp"):
            if self._playing:
                self._pause()
            else:
                self._play()
        imgui.same_line()

        if imgui.button(f"{fa.ICON_FA_STOP}  Stop##exp"):
            self._stop()
        imgui.same_line()

        imgui.text_disabled("Time:")
        imgui.same_line()
        imgui.text(self._time_label())
        imgui.same_line()

        seek_width = max(180, imgui.get_content_region_avail().x)
        imgui.set_next_item_width(seek_width)
        changed, val = imgui.slider_float("##seek_expanded", self._play_pos, 0.0, 1.0, format="")
        if changed:
            self._seek(val)

        imgui.spacing()

        # -- Volume row --
        imgui.text_disabled("Volume:")
        imgui.same_line()
        vol = self._state.audio_volume
        if vol <= 0.01:
            vol_icon = fa.ICON_FA_VOLUME_XMARK
        elif vol < 0.5:
            vol_icon = fa.ICON_FA_VOLUME_LOW
        else:
            vol_icon = fa.ICON_FA_VOLUME_HIGH
        imgui.text(vol_icon)
        imgui.same_line()
        imgui.set_next_item_width(max(140, imgui.get_content_region_avail().x - 80))
        vol_pct = vol * 100.0
        changed, new_pct = imgui.slider_float("##vol_exp", vol_pct, 0.0, 100.0, format="%.0f%%")
        if changed:
            self._state.audio_volume = new_pct / 100.0

        imgui.spacing()

        # -- Output device row --
        imgui.text_disabled("Output Device:")
        imgui.same_line()
        imgui.set_next_item_width(max(220, imgui.get_content_region_avail().x - 120))
        changed, self._out_device_idx = imgui.combo(
            "##out_dev", self._out_device_idx, self._output_devices
        )
        imgui.same_line()

        if imgui.button(f"{fa.ICON_FA_CHEVRON_DOWN}  Collapse"):
            self._state.audio_panel_expanded = False
