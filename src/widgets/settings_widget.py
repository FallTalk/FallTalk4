from PySide6.QtWidgets import QStackedWidget, QWidget
from qfluentwidgets import SegmentedWidget

from src.widgets.falltalk_widget import FallTalkWidget
from src.settings.main_settings import FallTalkSettings
from src.settings.rvc_settings import RVCSettings
from src.settings.xtts_settings import XTTSSettings
from src.settings.gpt_sovits_settings import GPTSoVITSSettings
from src.settings.styletts2_settings import StyleTTS2Settings

class SettingsWidget(FallTalkWidget):
    """
    Widget for displaying and managing application settings.
    """
    def __init__(self, parent=None):
        super().__init__(text="Settings", parent=parent, vertical=True)
        
        self.segmented_widget = SegmentedWidget(self)
        self.segmented_widget.setObjectName("settingsSegmentedWidget")
        
        self.stackedWidget = QStackedWidget(self)
        self.stackedWidget.setObjectName("settingsStackedWidget")
        
        self.main_settings = FallTalkSettings(self)
        self.rvc_settings = RVCSettings(self)
        self.xtts_settings = XTTSSettings(self)
        self.gpt_sovits_settings = GPTSoVITSSettings(self)
        self.styletts2_settings = StyleTTS2Settings(self)
        
        self.addSubInterface(self.main_settings, "mainSettings", "Main")
        self.addSubInterface(self.rvc_settings, "rvcSettings", "RVC")
        self.addSubInterface(self.xtts_settings, "xttsSettings", "XTTS")
        self.addSubInterface(self.gpt_sovits_settings, "gptSovitsSettings", "GPT-SoVITS")
        self.addSubInterface(self.styletts2_settings, "styletts2Settings", "StyleTTS2")
        
        self.segmented_widget.setCurrentItem("mainSettings")
        self.segmented_widget.currentItemChanged.connect(self.onCurrentIndexChanged)
        
        self.main_layout.addWidget(self.segmented_widget)
        self.main_layout.addWidget(self.stackedWidget)

    def addSubInterface(self, widget: QWidget, objectName, text):
        """Add sub interface to settings widget"""
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.segmented_widget.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget)
        )

    def onCurrentIndexChanged(self, index):
        """Handle change of current settings tab"""
        widget = self.stackedWidget.widget(self.stackedWidget.currentIndex())
        self.stackedWidget.setCurrentWidget(widget)