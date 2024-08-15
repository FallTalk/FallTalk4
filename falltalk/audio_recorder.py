import os.path
import uuid
from datetime import datetime

from PySide6.QtCore import QUrl, Signal
# coding:utf-8
from PySide6.QtCore import Qt, QSize, QPropertyAnimation
from PySide6.QtGui import QPainter, QColor
from PySide6.QtMultimedia import QMediaRecorder, QMediaCaptureSession, QAudioInput, QMediaDevices, QMediaFormat
from PySide6.QtWidgets import QWidget, QGraphicsOpacityEffect, QHBoxLayout, QVBoxLayout
from qfluentwidgets import TransparentToolButton, ToolTipFilter, CaptionLabel, isDarkTheme, FluentStyleSheet, ComboBox, IndeterminateProgressBar

from falltalk.icons import FallTalkIcons
from falltalk.config import cfg


class MediaPlayBarButton(TransparentToolButton):
    """ Media play bar button """

    def _postInit(self):
        super()._postInit()
        self.installEventFilter(ToolTipFilter(self, 1000))
        self.setFixedSize(30, 30)
        self.setIconSize(QSize(18, 18))


class RecordButton(MediaPlayBarButton):
    """ Record button """

    def _postInit(self):
        super()._postInit()
        self.setIconSize(QSize(20, 20))
        self.setRecording(False)

    def setRecording(self, isRecording: bool):
        if isRecording:
            self.setIcon(FallTalkIcons.RECORD.icon(color=QColor.fromRgb(235, 0, 0, )))
            self.setToolTip(self.tr('Stop Recording'))
        else:
            self.setIcon(FallTalkIcons.RECORD.icon(color=QColor.fromRgb(120, 0, 0, )))
            self.setToolTip(self.tr('Record'))

class AudioRecorderBarBase(QWidget):
    """ Audio recorder bar base class """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.recorder = QMediaRecorder(self)
        self.capture_session = QMediaCaptureSession(self)
        self.audio_input = QAudioInput(self)
        self.device_combo = ComboBox(self)
        self.capture_session.setAudioInput(self.audio_input)
        self.capture_session.setRecorder(self.recorder)
        self.output_file = None

        self.populate_device_combo()
        self.device_combo.currentIndexChanged.connect(self.set_audio_device)

        self.recordButton = RecordButton(self)
        self.inProgressBar = IndeterminateProgressBar(self)
        self.inProgressBar.stop()
        #self.progressSlider = Slider(Qt.Orientation.Horizontal, self)

        self.opacityEffect = QGraphicsOpacityEffect(self)
        self.opacityAni = QPropertyAnimation(self.opacityEffect, b'opacity')
        self.opacityEffect.setOpacity(1)
        self.opacityAni.setDuration(250)

        self.setGraphicsEffect(self.opacityEffect)
        FluentStyleSheet.MEDIA_PLAYER.apply(self)

        self.recorder.recorderStateChanged.connect(self._onStatusChanged)
        self.recordButton.clicked.connect(self.toggleRecording)

    def startRecording(self):
        self.output_file = QUrl.fromLocalFile(self.get_output_file())
        self.recorder.setMediaFormat(QMediaFormat.FileFormat.Wave)
        self.recorder.setAudioSampleRate(40000)
        self.recorder.setAudioBitRate(128000)
        self.recorder.setAudioChannelCount(1)
        self.recorder.setQuality(QMediaRecorder.Quality.VeryHighQuality)
        self.recorder.setOutputLocation(self.output_file)
        self.recorder.record()
        self.inProgressBar.start()

    def get_output_file(self):
        unique_id = uuid.uuid4()
        current_time = datetime.now()
        time_stamp = current_time.strftime("%Y%m%d%H%M%S")
        file_name = f"{unique_id.hex[:10]}_{time_stamp}"

        path = os.path.abspath(os.path.join(cfg.get(cfg.output_dir), "recordings", f"{file_name}.wav"))
        os.makedirs(os.path.join(cfg.get(cfg.output_dir), "recordings"), exist_ok=True)
        return path

    def stopRecording(self):
        self.inProgressBar.stop()
        self.recorder.stop()

    def populate_device_combo(self):
        audio_devices = QMediaDevices.audioInputs()
        for device in audio_devices:
            self.device_combo.addItem(device.description(), userData=device)

    def set_audio_device(self, index):
        device = self.device_combo.itemData(index)
        self.audio_input.setDevice(device)

    def _onStatusChanged(self, status):
        self.recordButton.setRecording(status == QMediaRecorder.RecorderState.RecordingState)

    def toggleRecording(self):
        """ toggle the recording state of audio recorder """
        if self.recorder.recorderState() == QMediaRecorder.RecorderState.StoppedState:
            self.startRecording()
            self.recordButton.setRecording(True)
        else:
            self.stopRecording()
            self.recordButton.setRecording(False)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing)

        if isDarkTheme():
            painter.setBrush(QColor(46, 46, 46))
            painter.setPen(QColor(0, 0, 0, 20))
        else:
            painter.setBrush(QColor(248, 248, 248))
            painter.setPen(QColor(0, 0, 0, 10))

        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)

class StandardAudioRecorderBar(AudioRecorderBarBase):
    """ Standard audio recorder bar """

    doneRecording = Signal(object)


    def __init__(self, parent=None):
        super().__init__(parent)
        self.vBoxLayout = QVBoxLayout(self)
        self.timeLayout = QHBoxLayout()
        self.buttonLayout = QHBoxLayout()
        self.leftButtonContainer = QWidget()
        self.centerButtonContainer = QWidget()
        self.rightButtonContainer = QWidget()
        self.leftButtonLayout = QHBoxLayout(self.leftButtonContainer)
        self.centerButtonLayout = QHBoxLayout(self.centerButtonContainer)
        self.rightButtonLayout = QHBoxLayout(self.rightButtonContainer)

        self.currentTimeLabel = CaptionLabel('0:00:00', self)

        self.__initWidgets()

    def __initWidgets(self):
        self.setFixedHeight(82)
        self.vBoxLayout.setSpacing(6)
        self.vBoxLayout.setContentsMargins(5, 9, 5, 9)

        self.vBoxLayout.addWidget(self.inProgressBar, 1, Qt.AlignmentFlag.AlignTop)

        self.vBoxLayout.addStretch(1)
        self.vBoxLayout.addLayout(self.buttonLayout, 1)
        self.buttonLayout.setContentsMargins(0, 0, 0, 0)
        self.leftButtonLayout.setContentsMargins(4, 0, 0, 0)
        self.centerButtonLayout.setContentsMargins(0, 0, 0, 0)
        self.rightButtonLayout.setContentsMargins(0, 0, 4, 0)
        self.rightButtonLayout.addWidget(self.device_combo)
        self.leftButtonLayout.addWidget(self.currentTimeLabel)

        self.centerButtonLayout.addWidget(self.recordButton)

        self.buttonLayout.addWidget(self.leftButtonContainer, 0, Qt.AlignmentFlag.AlignLeft)
        self.buttonLayout.addWidget(self.centerButtonContainer, 0, Qt.AlignmentFlag.AlignHCenter)
        self.buttonLayout.addWidget(self.rightButtonContainer, 0, Qt.AlignmentFlag.AlignRight)

        self.recorder.durationChanged.connect(self._onDurationChanged)

    def _onStatusChanged(self, status):
        super()._onStatusChanged(status)
        if status == QMediaRecorder.RecorderState.RecordingState:
            self.currentTimeLabel.setText('0:00:00')
        else:
            self.doneRecording.emit(self.output_file.toLocalFile())

    def _onDurationChanged(self, dur):
        self.currentTimeLabel.setText(self._formatTime(self.recorder.duration()))

    def _formatTime(self, time: int):
        time = int(time / 1000)
        s = time % 60
        m = int(time / 60)
        h = int(time / 3600)
        return f'{h}:{m:02}:{s:02}'
