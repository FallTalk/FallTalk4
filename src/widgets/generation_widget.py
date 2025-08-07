from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QGroupBox, QHBoxLayout
from qfluentwidgets import TextEdit, FluentIcon as FIF, RangeSettingCard, SwitchSettingCard, ConfigValidator, \
    ConfigItem, ToolButton

from src.config.config import cfg
from src.ui.cards import TextSettingCard, RangeSettingCardScaled
from src.utils.icons import FallTalkIcons
from src.widgets.falltalk_widget import FallTalkWidget
from src.widgets.drawer import RightDrawer

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
            self.tr('Use RVC Upscaler (Recommended For Untrained)'),
            cfg.rvc_enabled
        )

        self.delete_leftovers = SwitchSettingCard(
            FIF.DELETE,
            self.tr('Keep Only FUZ'),
            self.tr('Delete XMW, LIP, and WAV'),
            cfg.keep_only_fuz
        )

        self.upscaler_enabled = SwitchSettingCard(
            FIF.MEGAPHONE,
            self.tr('Super Resolution'),
            self.tr('Use Super Resolution Upscaler (Recommended)'),
            cfg.apbwe_enabled
        )

        self.pad_short_phrases = SwitchSettingCard(
            FallTalkIcons.PADDING.icon(stroke=True),
            self.tr('Pad Short Phrases'),
            self.tr('Duplicate short phrases to improve quality, increases generation time'),
            cfg.pad_short_phrases
        )

        self.gen_settings = QGroupBox()
        self.gen_settings.setStyleSheet("border: none")
        self.gen_settings_layout = QHBoxLayout()
        self.gen_settings_layout.setContentsMargins(0, 0, 0, 0)

        self.gen_settings_layout.addWidget(self.autoplay, 2)
        self.gen_settings_layout.addWidget(self.rvc_enabled, 2)
        self.gen_settings_layout.addWidget(self.upscaler_enabled, 2)
        self.gen_settings_layout.addWidget(self.pad_short_phrases, 2)
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

        self.help_drawer = RightDrawer(self, title="About", icon=FIF.QUESTION)
        self.settings_drawer = RightDrawer(self, title="Advanced Settings", icon=FIF.SETTING)

        self.settings_button = ToolButton()
        self.settings_button.setIcon(FIF.SETTING)
        self.settings_button.setEnabled(True)
        self.settings_button.clicked.connect(lambda: self.toggle_settings_drawer())
        self.settings_button.setFixedWidth(50)

        self.help_button = ToolButton()
        self.help_button.setIcon(FIF.QUESTION)
        self.help_button.setEnabled(True)
        self.help_button.clicked.connect(lambda: self.toggle_help_drawer())
        self.help_button.setFixedWidth(50)


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

    def toggle_settings_drawer(self):
        self.settings_drawer.open_drawer()

    def toggle_help_drawer(self):
        self.help_drawer.open_drawer()
