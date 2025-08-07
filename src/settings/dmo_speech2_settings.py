from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon as FIF, RangeSettingCard, SettingCardGroup, isDarkTheme
)
from qfluentwidgets import ScrollArea, ExpandLayout

from src.config.config import cfg
from ui.cards import RangeSettingCardScaled


class DMSpeech2Settings(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr(''), self.scroll_widget)

        self.temperature_card = RangeSettingCardScaled(
            cfg.dmo_speech2_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Control randomness in generation'),
            parent=self.settings_group
        )

        self.teacher_steps_card = RangeSettingCard(
            cfg.dmo_speech2_teacher_steps,
            FIF.BOOK_SHELF,
            self.tr('Teacher Steps'),
            self.tr('Number of steps for the teacher model'),
            parent=self.settings_group
        )

        self.teacher_stopping_time_card = RangeSettingCardScaled(
            cfg.dmo_speech2_teacher_stopping_time,
            FIF.STOP_WATCH,
            self.tr('Teacher Stopping Time'),
            self.tr('Stopping time for the teacher model'),
            parent=self,
            scale=100.0
        )

        self.student_start_step_card = RangeSettingCard(
            cfg.dmo_speech2_student_start_step,
            FIF.PLAY,
            self.tr('Student Start Step'),
            self.tr('Starting step for the student model'),
            parent=self
        )

        self.__initWidget()

    def __initWidget(self):
        self.resize(1000, 800)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 0, 0, 20)
        self.setWidget(self.scroll_widget)
        self.setWidgetResizable(True)

        # initialize style sheet
        self.__setQss()

        # initialize layout
        self.__initLayout()
        self.__connectSignalToSlot()

    def __initLayout(self):
        # add cards to group
        self.settings_group.addSettingCard(self.temperature_card)
        self.settings_group.addSettingCard(self.teacher_steps_card)
        self.settings_group.addSettingCard(self.teacher_stopping_time_card)
        self.settings_group.addSettingCard(self.student_start_step_card)

        # add setting card group to layout
        self.expand_layout.setSpacing(28)
        self.expand_layout.setContentsMargins(15, 0, 15, 0)
        self.expand_layout.addWidget(self.settings_group)

    def __setQss(self):
        """ set style sheet """
        self.scroll_widget.setObjectName('scrollWidget')

        theme = 'dark' if isDarkTheme() else 'light'
        with open(f'resource/qss/{theme}/setting_interface.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def __connectSignalToSlot(self):
        pass