from qfluentwidgets import (
    FluentIcon as FIF, RangeSettingCard
)

from src.config.config import cfg
from src.settings.generic_settings import GenericSettings
from ui.cards import RangeSettingCardScaled


class DMSpeech2Settings(GenericSettings):
    def __init__(self, parent=None):
        super().__init__(parent)

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
        # add cards to group
        self.settings_group.addSettingCard(self.temperature_card)
        self.settings_group.addSettingCard(self.teacher_steps_card)
        self.settings_group.addSettingCard(self.teacher_stopping_time_card)
        self.settings_group.addSettingCard(self.student_start_step_card)

        self.setupLayout()