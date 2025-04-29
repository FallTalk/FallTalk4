from PySide6.QtMultimedia import QMediaDevices
from qfluentwidgets import ComboBox
from qfluentwidgets.multimedia import StandardMediaPlayBar


class StandardAudioPlayerBar(StandardMediaPlayBar):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.device_combo = ComboBox(self)
        self.rightButtonLayout.addWidget(self.device_combo)
        self.populate_device_combo()
        self.device_combo.currentIndexChanged.connect(self.set_audio_device)

    def populate_device_combo(self):
        audio_devices = QMediaDevices.audioOutputs()
        for device in audio_devices:
            self.device_combo.addItem(device.description(), userData=device)

    def set_audio_device(self, index):
        device = self.device_combo.itemData(index)
        self.player._audioOutput.setDevice(device)
