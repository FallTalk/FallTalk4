from __future__ import annotations

from typing import TYPE_CHECKING

from src.enums.engine_type import EngineType

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QStackedWidget, QWidget
from qfluentwidgets import SegmentedWidget

from src.widgets.falltalk_widget import FallTalkWidget
from src.settings.main_settings import FallTalkSettings
from src.settings.rvc_settings import RVCSettings


class SettingsWidget(FallTalkWidget):

    def __init__(self, parent: FallTalkApp):
        super().__init__(parent=parent, text="Settings", vertical=True)

        # Create a TabView instance
        self.pivot = SegmentedWidget(self)
        self.stackedWidget = QStackedWidget(self)

        # Add TabItems to the TabView
        self.rvc_settings = RVCSettings(parent)
        # self.gpt_sovits_settings = GPTSoVITSSettings(parent)
        # self.styletts2_settings = StyleTTS2Settings(parent)
        # self.f5_settings = F5Settings(parent)
        # self.fish_speech_settings = FishSpeechSettings(parent)
        # self.llasa_settings = LLASASettings(parent)
        # self.orpheus_settings = OrpheusSettings(parent)

        self.engine_settings = FallTalkSettings(parent)

        # add items to pivot
        self.addSubInterface(self.engine_settings, 'main_settings', 'Main Settings')
        self.addSubInterface(self.rvc_settings, 'rvc_settings', EngineType.RVC.value)
        # self.addSubInterface(self.styletts2_settings, 'styletts2_settings', EngineType.STYLE_TTS2.value)
        # self.addSubInterface(self.gpt_sovits_settings, 'gpt_sovits_settings', EngineType.GPT_SOVITS.value)
        # self.addSubInterface(self.f5_settings, 'f5_settings', EngineType.F5.value)
        # self.addSubInterface(self.fish_speech_settings, 'fish_speech_settings', EngineType.FISH_SPEECH.value)
        # self.addSubInterface(self.llasa_settings, 'llasa_settings', EngineType.LLASA.value)
        # self.addSubInterface(self.orpheus_settings, 'orpheus_settings', EngineType.ORPHEUS.value)

        self.boxLayout.addWidget(self.pivot, 0, Qt.AlignmentFlag.AlignLeft)
        self.boxLayout.addWidget(self.stackedWidget)

        self.stackedWidget.currentChanged.connect(self.onCurrentIndexChanged)
        self.stackedWidget.setCurrentWidget(self.engine_settings)
        self.pivot.setCurrentItem(self.engine_settings.objectName())

    def addSubInterface(self, widget: QWidget, objectName, text):
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget))
        # Create settings buttons for each engine

    def onCurrentIndexChanged(self, index):
        widget = self.stackedWidget.widget(index)
        self.pivot.setCurrentItem(widget.objectName())
