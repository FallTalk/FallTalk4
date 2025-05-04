from PySide6.QtCore import Qt
from PySide6.QtWidgets import QStackedWidget, QWidget
from qfluentwidgets import SegmentedWidget

from src.widgets.falltalk_widget import FallTalkWidget
from src.settings.main_settings import FallTalkSettings
from src.settings.rvc_settings import RVCSettings
from src.settings.xtts_settings import XTTSSettings
from src.settings.gpt_sovits_settings import GPTSoVITSSettings
from src.settings.styletts2_settings import StyleTTS2Settings

class SettingsWidget(FallTalkWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent, text="Settings", vertical=True)

        # Create a TabView instance
        self.pivot = SegmentedWidget(self)
        self.stackedWidget = QStackedWidget(self)

        # Add TabItems to the TabView
        self.rvc_settings = RVCSettings(parent)
        self.xtts_settings = XTTSSettings(parent)
        self.gpt_sovits_settings = GPTSoVITSSettings(parent)
        self.styletts2_settings = StyleTTS2Settings(parent)

        self.engine_settings = FallTalkSettings(parent)

        # add items to pivot
        self.addSubInterface(self.engine_settings, 'main_settings', 'Main Settings')
        self.addSubInterface(self.rvc_settings, 'rvc_settings', 'RVC')
        self.addSubInterface(self.xtts_settings, 'xtts_settings', 'XTTS')
        self.addSubInterface(self.gpt_sovits_settings, 'gpt_sovits_settings', 'GPT SoVITS')
        self.addSubInterface(self.styletts2_settings, 'styletts2_settings', 'StyleTTS2')

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
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget)
        )

    def onCurrentIndexChanged(self, index):
        widget = self.stackedWidget.widget(index)
        self.pivot.setCurrentItem(widget.objectName())