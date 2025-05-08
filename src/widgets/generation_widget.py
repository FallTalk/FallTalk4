from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

from PySide6.QtCore import QPoint
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QGroupBox, QHBoxLayout
from qfluentwidgets import TextEdit, FluentIcon as FIF, RangeSettingCard, SwitchSettingCard, ConfigValidator, ConfigItem, Flyout, FlyoutAnimationType, FlyoutView, ScrollArea

from src.config.config import cfg
from src.ui.cards import TextSettingCard, RangeSettingCardScaled
from src.utils.icons import FallTalkIcons
from src.widgets.falltalk_widget import FallTalkWidget


class GenerationWidget(FallTalkWidget):

    def __init__(self, text: str, parent: 'FallTalkApp') -> None:
        super().__init__(parent=parent, text=text, vertical=True)
        self.text_input = TextEdit()
        font = QFont()
        font.setPointSize(12)
        self.text_input.setFont(font)
        self.addToFrame(self.text_input)
        self.parent = parent
        self.gen_settings = None
        self.gen_settings2 = None
        self.media_player = None
        self.generate_button = None

    def addGenSettings(self):
        self.output_name = ConfigItem("TTS", "output_name", None, ConfigValidator())

        self.autoplay = SwitchSettingCard(
            FIF.PLAY,
            self.tr('Autoplay'),
            self.tr('Automatically Play Generated Audio'),
            cfg.auto_play,
        )
        self.output_name_card = TextSettingCard(
            self.output_name,
            FIF.SAVE_AS,
            self.tr('Output Name'),
            self.tr('Name of Generated WAV file'),
            placeholder="Random"
        )
        self.xwm_card = SwitchSettingCard(
            FIF.COMMAND_PROMPT,
            self.tr('Create FUZ'),
            self.tr('Create XWM, LIP, and FUZ'),
            cfg.xwm_enabled,
        )
        self.rvc_enabled = SwitchSettingCard(
            FIF.MEGAPHONE,
            self.tr('RVC'),
            self.tr('Use RVC Upscaler (Recommended)'),
            cfg.rvc_enabled
        )

        self.delete_leftovers = SwitchSettingCard(
            FIF.DELETE,
            self.tr('Keep Only FUZ'),
            self.tr('Delete XMW, LIP, and WAV'),
            cfg.keep_only_fuz
        )

        self.gen_settings = QGroupBox()
        self.gen_settings.setStyleSheet("border: none")
        self.gen_settings_layout = QHBoxLayout()
        self.gen_settings_layout.setContentsMargins(0, 0, 0, 0)

        self.gen_settings_layout.addWidget(self.autoplay, 2)
        self.gen_settings_layout.addWidget(self.rvc_enabled, 2)
        self.gen_settings.setLayout(self.gen_settings_layout)

        self.gen_settings2 = QGroupBox()
        self.gen_settings2.setStyleSheet("border: none")
        self.gen_settings2_layout = QHBoxLayout()
        self.gen_settings2_layout.setContentsMargins(0, 0, 0, 0)

        self.gen_settings2_layout.addWidget(self.xwm_card, 2)
        self.gen_settings2_layout.addWidget(self.delete_leftovers, 2)
        self.gen_settings2.setLayout(self.gen_settings2_layout)

        self.addToFrame(self.output_name_card)
        self.addToFrame(self.gen_settings2)
        self.addToFrame(self.gen_settings)

    def addTempAndRep(self):
        self.temperature_card = RangeSettingCardScaled(
            cfg.model_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Randomness, 1 = balanced, 0 = disabled'),
        )
        self.repetition_penalty_card = RangeSettingCard(
            cfg.model_repetition,
            FallTalkIcons.LOOP.icon(),
            self.tr('Repetition Penalty'),
            self.tr('Discourage same sounds during generation'),
        )
        self.temp_and_rep = QGroupBox()
        self.temp_and_rep.setStyleSheet("border: none")
        self.temp_and_rep_layout = QHBoxLayout()
        self.temp_and_rep_layout.setContentsMargins(0, 0, 0, 0)
        self.temp_and_rep_layout.addWidget(self.repetition_penalty_card, 3)
        self.temp_and_rep_layout.addWidget(self.temperature_card, 3)
        self.temp_and_rep.setLayout(self.temp_and_rep_layout)
        self.addToFrame(self.temp_and_rep)

    def addTempAndStopRep(self):
        # self.temperature_card = RangeSettingCardScaled(
        #     cfg.voicecraft_temperature,
        #     FIF.FRIGID,
        #     self.tr('Temperature'),
        #     self.tr('Randomness, 1 = balanced, 0 = disabled'),
        # )
        self.repetition_penalty_card = RangeSettingCard(
            cfg.stop_repetition,
            FallTalkIcons.LOOP.icon(),
            self.tr('Stop Repetition'),
            self.tr('If Long Pauses, change to 2 or 1. -1 = disabled'),
        )
        self.temp_and_rep = QGroupBox()
        self.temp_and_rep.setStyleSheet("border: none")
        self.temp_and_rep_layout = QHBoxLayout()
        self.temp_and_rep_layout.setContentsMargins(0, 0, 0, 0)
        self.temp_and_rep_layout.addWidget(self.repetition_penalty_card, 3)
        # self.temp_and_rep_layout.addWidget(self.temperature_card, 3)
        self.temp_and_rep.setLayout(self.temp_and_rep_layout)
        self.addToFrame(self.temp_and_rep)



    def show_flyout(self, widget, icon, title):
        view = FlyoutView(
            title=title,
            content="",
            icon=icon,
            parent=self,
            isClosable=True
        )

        # Add settings widget
        view.vBoxLayout.addWidget(widget)
        window_rect = self.window().geometry()

        # Determine and set flyout size explicitly
        width = max(1000, int(window_rect.width() * 0.75))
        height = max(600, int(window_rect.height() * 0.75))
        view.resize(width, height)

        # Calculate dynamic offsets based on window size
        x_offset = -(window_rect.width() * 0.3)  # Move left by 30% of window width
        y_offset = -(window_rect.height() * 0.2)  # Move up by 20% of window height

        # Calculate center point using explicit width/height
        center_point = QPoint(
            int(window_rect.x() + (window_rect.width() - width) // 2 + x_offset),
            int(window_rect.y() + (window_rect.height() - height) // 2 + y_offset)
        )

        # Show the flyout at the center point
        w = Flyout.make(view, center_point, self, aniType=FlyoutAnimationType.NONE)
        view.closed.connect(w.close)

    def show_settings(self, settings):
        self.show_flyout(settings,FIF.SETTING,'Advanced Settings')

    def show_help(self, help):
        self.show_flyout(help,FIF.QUESTION,'Help')