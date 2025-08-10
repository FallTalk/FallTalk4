from __future__ import annotations

import os
import shutil
import threading
from typing import TYPE_CHECKING

import PySide6

from src.ui.cards import TextSettingCard, SpinSettingCard
from src.utils.icons import FallTalkIcons
from src.widgets import RightDrawer
from src.widgets.generation_widget import GenerationWidget
from src.widgets.engine_widgets_config import SETTINGS_WIDGETS, HELP_WIDGETS

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

from PySide6.QtCore import Qt, QMetaObject, Q_ARG, QUrl, Signal, Slot
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QFrame, QGroupBox, QRadioButton, QButtonGroup
from qfluentwidgets import (
    PrimaryPushButton, FluentIcon as FIF,
    ToolButton, isDarkTheme, SingleDirectionScrollArea, SwitchSettingCard, ConfigValidator,
    ConfigItem, PushButton, Flyout, InfoBarIcon
)

from src.audio.audio_player import StandardAudioPlayerBar
from src.config.config import cfg
from src.utils.logging_utils import logger
from src.utils.file_utils import formatted_time_stamp_uuid, get_output_file_name
from src.utils.filesystem_utils import get_app_root
from src.enums.engine_type import EngineType
from src.utils.inference_utils import get_default_reference_and_transcript, generic_inference, preprocess_text
from src.utils.audio_utils import combine_wav_files, combine_references


class GeneratedAudioItem(QFrame):
    """A widget representing a single generated audio item."""

    # Signal to notify when an item is saved
    save_requested = Signal(str)  # audio_file

    def __init__(self, index: int, audio_file: str, parent=None, output_name=None):
        super().__init__(parent)
        self.index = index
        self.audio_file = audio_file
        self.output_name = output_name

        # Setup UI
        self.setObjectName("GeneratedAudioItem")
        self.updateStyle()

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Header layout with index and radio button
        self.header_layout = QHBoxLayout()

        # Radio button for selection
        self.radio_button = QRadioButton(f"Generation #{index + 1}", self)
        self.header_layout.addWidget(self.radio_button)

        # Play button
        self.play_button = ToolButton(self)
        self.play_button.setIcon(FIF.PLAY)
        self.play_button.setToolTip("Play audio")
        self.play_button.clicked.connect(self.play_audio)
        self.header_layout.addWidget(self.play_button, alignment=Qt.AlignRight)

        # Save button
        self.save_button = ToolButton(self)
        self.save_button.setIcon(FIF.SAVE)
        self.save_button.setToolTip("Save this generation")
        self.save_button.clicked.connect(self.save_audio)
        self.header_layout.addWidget(self.save_button, alignment=Qt.AlignRight)

        self.header_layout.addStretch()
        self.layout.addLayout(self.header_layout)

        # Audio player
        self.audio_player = StandardAudioPlayerBar(self)
        self.audio_player.setVolume(100)
        self.audio_player.setVisible(False)  # Hide until play is clicked
        self.layout.addWidget(self.audio_player)

        # Set audio file
        if audio_file and os.path.exists(audio_file):
            self.audio_player.player.setSource(QUrl.fromLocalFile(audio_file))

    def updateStyle(self):
        """Update the style based on the current theme."""
        if isDarkTheme():
            bg_color = "#2D2D30"  # Dark gray for dark theme
        else:
            bg_color = "#F5F5F5"  # Light gray for light theme

        self.setStyleSheet(
            "QFrame#GeneratedAudioItem { "
            f"background-color: {bg_color}; "
            "border-radius: 10px; "
            "padding: 10px; "
            "margin: 5px; "
            "}"
        )

    def play_audio(self):
        """Play the audio file."""
        if self.audio_file and os.path.exists(self.audio_file):
            self.audio_player.player.stop()
            self.audio_player.player.setSource(QUrl.fromLocalFile(self.audio_file))
            self.audio_player.player.play()
            self.audio_player.setVisible(True)

    def save_audio(self):
        """Save this audio generation."""
        if self.audio_file and os.path.exists(self.audio_file):
            # Emit signal to save this item
            self.save_requested.emit(self.audio_file)


class GenericMultiGenerationWidget(GenerationWidget):
    """
    A widget that allows generating multiple TTS outputs from the same text.
    It displays the results in a scrollable list with play buttons and a save button.
    """

    # Signal for when audio generation is complete
    generation_complete = Signal(list)  # List of audio files

    def __init__(self, parent: 'FallTalkApp'):
        super().__init__(parent=parent, text="Multi Generation")
        self.engine_type = None  # Will be set by update_engine_type
        self.text_input.setPlaceholderText("If no reference is selected, a default will be used. Selecting a reference audio can help change the emotion of the generated speech. ")

        # Remove the media player that was added by GenericGenerationWidget
        if self.media_player:
            self.media_player.setParent(None)
            self.boxLayout.removeWidget(self.media_player)

        # List to store generated audio files
        self.generated_audio_files = []
        self.audio_items = []
        self.button_group = QButtonGroup(self)

        # Connect signals
        self.generation_complete.connect(self.update_ui_with_generated_files)

        # Setup UI for multi-generation
        self.setup_multi_generation_ui()
        self.addGenSettings()

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout_widget = QGroupBox()

        self.generate_button = PrimaryPushButton(text="Generate Multiple")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.generate_multiple)

        # Clear chat button
        self.clear_button = PushButton(text="Clear Chat")
        self.clear_button.setIcon(FIF.DELETE)
        self.clear_button.clicked.connect(self.clear_results)

        # Add buttons to layout
        self.buttons_layout.addWidget(self.clear_button)
        self.buttons_layout.addWidget(self.generate_button)
        self.buttons_layout.addWidget(self.settings_button)
        self.buttons_layout.addWidget(self.help_button)

        self.boxLayout.addLayout(self.buttons_layout)

        self.setEnabled(False)


    def addGenSettings(self):
        self.output_name = ConfigItem("TTS", "output_name", None, ConfigValidator())

        self.threads_card = SpinSettingCard(
            cfg.multigen_total,
            FIF.STOP_WATCH,
            self.tr('Total Generations'),
            self.tr('How many?'),
            step=1
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


        self.gen_settings3 = QGroupBox()
        self.gen_settings3.setStyleSheet("border: none")
        self.gen_settings3_layout = QHBoxLayout()
        self.gen_settings3_layout.setContentsMargins(0, 0, 0, 0)
        self.gen_settings3_layout.addWidget(self.threads_card, 2)
        self.gen_settings3_layout.addWidget(self.output_name_card, 2)
        self.gen_settings3.setLayout(self.gen_settings3_layout)

        self.addToFrame(self.gen_settings3)
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


    def update_engine_type(self, engine_type: EngineType):
        """Update the widget's engine type."""
        self.engine_type = engine_type
        self.clear_results()

        # Add settings and help widgets based on engine type
        if engine_type in SETTINGS_WIDGETS:
            self.settings_widget = SETTINGS_WIDGETS[engine_type](self)
            self.settings_drawer.addWidget(self.settings_widget)

        if engine_type in HELP_WIDGETS:
            self.help_widget = HELP_WIDGETS[engine_type](self)
            self.help_drawer.addWidget(self.help_widget)

    def setup_multi_generation_ui(self):
        """Setup the UI for multi-generation."""

        # Create a scroll area for displaying generated audio items
        self.scroll_area = SingleDirectionScrollArea(orient=Qt.Vertical, parent=self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setSpacing(10)
        self.scroll_area.setWidget(self.scroll_content)

        # Add scroll area to main layout
        self.addToFrame(self.scroll_area)

    def generate_multiple(self):
        """Generate multiple TTS outputs from the same text."""

        if not self.text_input.toPlainText().strip():
            self.showErrorPopup(self, self.generate_button, "Please Enter Some Text")
            return
        else:
            text = preprocess_text(self.text_input.toPlainText())

        current_engine = EngineType(cfg.get(cfg.engine))
        references = self.parent.reference_widget.reference_audio
        references_length = self.parent.reference_widget.reference_audio_length
        transcribe_state = None

        if current_engine.needs_reference_when_trained or self.parent.tts_engine.is_base:
            if references is None or not references:
                # Get default reference from default_references.json and characters.json
                character_name = self.parent.tts_engine.model_name
                reference_path, transcribe_state = get_default_reference_and_transcript(self.parent, character_name)

                # If reference found, use it
                if reference_path:
                    references = [reference_path]
                else:
                    self.showErrorPopup(self, self.generate_button, "Please Select Reference Audio")
                    return
            elif references_length > current_engine.max_reference_length or references_length < 3:
                self.showErrorPopup(self, self.generate_button,
                                    f"Please Select between 3 and {current_engine.max_reference_length} seconds of Reference Audio")
                return
            elif references_length < 3:
                self.showErrorPopup(self, self.generate_button,
                                    "Please Select at least 3 seconds of Reference Audio")
                return

        # Clear previous results
        self.clear_results()

        # Get number of generations
        num_generations = cfg.get(cfg.multigen_total)

        # Show loader
        QMetaObject.invokeMethod(self.parent, "showLoaderPopup", Qt.QueuedConnection,
                                Q_ARG(str, f"Generating Audio"),
                                Q_ARG(str, f"Generating {num_generations} audio samples..."))

        # Start generation in a separate thread
        threading.Thread(target=self.generate_audio_thread, args=(text, num_generations, references, transcribe_state), daemon=True).start()

    def generate_audio_thread(self, text: str, num_generations: int, references: [], transcribe_state: str):
        """Thread function to generate multiple audio files."""
        try:
            # Create output directory
            output_dir = os.path.join(get_app_root(), "output")
            os.makedirs(output_dir, exist_ok=True)

            # Clear previous files
            self.generated_audio_files = []

            # Generate audio files
            for i in range(num_generations):
                # Update loader message
                QMetaObject.invokeMethod(self.parent, "update_loader", Qt.QueuedConnection,
                                        Q_ARG(str, f"Generating sample {i+1} of {num_generations}"))

                # Create output file path
                file_name = f"multi_gen_{i}_{formatted_time_stamp_uuid()}"
                output_file = get_output_file_name(file_name, cfg.get(cfg.output_dir), self.parent.tts_engine.model_name, self.parent.tts_engine.engin_type.value )
                # Generate audio
                generic_inference(
                    self.parent,
                    output_file=output_file,
                    text=text,
                    selected_audio=combine_references(references),
                    panel=None,
                    transcribe_state=transcribe_state,
                    api=True
                )

                # Add to list of generated files
                self.generated_audio_files.append(output_file)

            # Emit signal to update UI with generated files
            self.generation_complete.emit(self.generated_audio_files)

            # Close loader
            QMetaObject.invokeMethod(self.parent, "close_loader", Qt.QueuedConnection,
                                    Q_ARG(PySide6.QtCore.QObject, self.parent))

        except Exception as e:
            logger.exception("Error generating multiple audio files")
            QMetaObject.invokeMethod(self.parent, "onError", Qt.QueuedConnection,
                                    Q_ARG(PySide6.QtCore.QObject, self.parent),
                                    Q_ARG(str, "Generation Error"),
                                    Q_ARG(str, f"Error generating audio: {str(e)}"))

            # Close loader
            QMetaObject.invokeMethod(self.parent, "close_loader", Qt.QueuedConnection,
                                    Q_ARG(PySide6.QtCore.QObject, self.parent))

    @Slot(list)
    def update_ui_with_generated_files(self, audio_files=None):
        """Update the UI with the generated audio files."""
        # Use provided audio files if available, otherwise use stored ones
        if audio_files:
            self.generated_audio_files = audio_files

        # Clear any existing items
        for item in self.audio_items:
            item.deleteLater()
        self.audio_items.clear()
        self.button_group = QButtonGroup(self)

        # Add new items
        for i, audio_file in enumerate(self.generated_audio_files):
            # Create item with output name from the widget
            output_name = self.output_name_card.configItem.value
            item = GeneratedAudioItem(i, audio_file, parent=self.scroll_content, output_name=output_name)

            # Connect the save_requested signal to handle_save_requested
            item.save_requested.connect(self.handle_save_requested)

            self.scroll_layout.addWidget(item)
            self.audio_items.append(item)
            self.button_group.addButton(item.radio_button, i)

        # Select the first item by default
        if self.audio_items:
            self.audio_items[0].radio_button.setChecked(True)

        # Scroll to the top
        self.scroll_area.verticalScrollBar().setValue(0)

    @Slot(str, str)
    def handle_save_requested(self, audio_file):
        """Handle save request from an audio item."""
        if not audio_file or not os.path.exists(audio_file):
            return

        # Get the output name from the widget if not provided
        output_name = self.output_name_card.configItem.value

        # Use a default name if still empty
        if not output_name:
            base, ext = os.path.splitext(audio_file)
            final_output_file = f"{base}_final{ext}"
        else:
            dir_name = os.path.dirname(audio_file)
            final_output_file = str(os.path.join(dir_name, output_name+".wav"))

        shutil.copy2(audio_file, final_output_file)


        #
        # # Update the media player in the parent widget
        # QMetaObject.invokeMethod(self.parent, "updateMediaplayer", Qt.QueuedConnection,
        #                         Q_ARG(PySide6.QtCore.QObject, self),
        #                         Q_ARG(str, final_output_file))

        # Create lip and fuz if enabled
        if cfg.get(cfg.xwm_enabled):
            from src.utils.audio_utils import create_lip_and_fuz
            create_lip_and_fuz(self.parent, final_output_file)

        # Clear results
        self.clear_and_delete_results()

        # Notify user
        QMetaObject.invokeMethod(self.parent, "afterGen", Qt.QueuedConnection,
                                Q_ARG(PySide6.QtCore.QObject, self.parent))


    def clear_and_delete_results(self):
        self.clear_results()

        # Clear UI items
        for item in self.audio_items:
            item.deleteLater()

    def clear_results(self):
        """Clear all generated results."""
        # Clear audio files list
        self.generated_audio_files = []


        self.audio_items.clear()

        # Clear the scroll layout
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def clear(self):
        """Clear the widget state."""
        self.clear_results()

    def showErrorPopup(self, parent, target, content):
        Flyout.create(
            icon=InfoBarIcon.ERROR,
            title='Error',
            content=content,
            target=target,
            parent=parent,
            isClosable=True
        )

