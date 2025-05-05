from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QStackedWidget, QWidget, QHBoxLayout
from qfluentwidgets import SegmentedWidget, PushButton, Flyout, FlyoutView, FlyoutAnimationType

from src.widgets.falltalk_widget import FallTalkWidget
from src.settings.main_settings import FallTalkSettings
from src.settings.rvc_settings import RVCSettings
from src.settings.xtts_settings import XTTSSettings
from src.settings.gpt_sovits_settings import GPTSoVITSSettings
from src.settings.styletts2_settings import StyleTTS2Settings
from src.settings.f5_settings import F5Settings
from src.settings.fish_speech_settings import FishSpeechSettings
from src.settings.llasa_settings import LLASASettings
from src.settings.orpheus_settings import OrpheusSettings

class SettingsWidget(FallTalkWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent, text="Settings", vertical=True)

        # Create a TabView instance
        self.pivot = SegmentedWidget(self)
        self.stackedWidget = QStackedWidget(self)

        # Add TabItems to the TabView
        self.engine_settings = FallTalkSettings(parent)

        # Create settings buttons for each engine
        self.rvc_settings_btn = PushButton("RVC Settings")
        self.xtts_settings_btn = PushButton("XTTS Settings")
        self.gpt_sovits_settings_btn = PushButton("GPT SoVITS Settings")
        self.styletts2_settings_btn = PushButton("StyleTTS2 Settings")
        self.f5_settings_btn = PushButton("F5 Settings")
        self.fish_speech_settings_btn = PushButton("FishSpeech Settings")
        self.llasa_settings_btn = PushButton("LLASA Settings")
        self.orpheus_settings_btn = PushButton("Orpheus Settings")

        # Connect buttons to show settings
        self.rvc_settings_btn.clicked.connect(lambda: self.show_settings(RVCSettings(self)))
        self.xtts_settings_btn.clicked.connect(lambda: self.show_settings(XTTSSettings(self)))
        self.gpt_sovits_settings_btn.clicked.connect(lambda: self.show_settings(GPTSoVITSSettings(self)))
        self.styletts2_settings_btn.clicked.connect(lambda: self.show_settings(StyleTTS2Settings(self)))
        self.f5_settings_btn.clicked.connect(lambda: self.show_settings(F5Settings(self)))
        self.fish_speech_settings_btn.clicked.connect(lambda: self.show_settings(FishSpeechSettings(self)))
        self.llasa_settings_btn.clicked.connect(lambda: self.show_settings(LLASASettings(self)))
        self.orpheus_settings_btn.clicked.connect(lambda: self.show_settings(OrpheusSettings(self)))

        # Add buttons to layout
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(self.rvc_settings_btn)
        self.buttons_layout.addWidget(self.xtts_settings_btn)
        self.buttons_layout.addWidget(self.gpt_sovits_settings_btn)
        self.buttons_layout.addWidget(self.styletts2_settings_btn)
        self.buttons_layout.addWidget(self.f5_settings_btn)
        self.buttons_layout.addWidget(self.fish_speech_settings_btn)
        self.buttons_layout.addWidget(self.llasa_settings_btn)
        self.buttons_layout.addWidget(self.orpheus_settings_btn)

        # Add main settings and buttons to layout
        self.boxLayout.addWidget(self.engine_settings)
        self.boxLayout.addLayout(self.buttons_layout)

    def show_settings(self, settings):
        view = FlyoutView(
            title='Settings',
            content="",
            icon=FIF.SETTING,
            parent=self,
            isClosable=True
        )

        # Add settings widget
        view.vBoxLayout.addWidget(settings)

        # Adjust flyout size
        screen_rect = self.window().screen().availableGeometry()
        width = min(1000, screen_rect.width() - 100)  # Leave some margin
        view.setMinimumWidth(width)

        # Calculate center point of the window
        window_rect = self.window().geometry()
        view_size = view.sizeHint()
        center_point = QPoint(
            window_rect.x() + window_rect.width() // 2 - view_size.width() // 2,
            window_rect.y() + window_rect.height() // 2 - view_size.height() // 2
        )

        # Show the flyout at the center point
        w = Flyout.make(view, center_point, self, aniType=FlyoutAnimationType.NONE)
        view.closed.connect(w.close)