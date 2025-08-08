from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Optional
import os
import threading

import PySide6
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.icons import FallTalkIcons
from widgets import RightDrawer

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

from PySide6.QtCore import Qt, Signal, QMetaObject, Q_ARG, QUrl, QTimer
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QScrollArea, QWidget, QLabel, QTextEdit, QFrame, QGroupBox
from qfluentwidgets import PrimaryPushButton, ScrollArea, FluentIcon as FIF, InfoBar, InfoBarPosition, isDarkTheme, \
    SwitchSettingCard, ToolButton, OptionsSettingCard, PushButton

from src.audio.audio_player import StandardAudioPlayerBar
from src.config.config import cfg
from src.utils.logging_utils import logger
from src.utils.file_utils import formatted_time_stamp_uuid
from src.utils.filesystem_utils import get_app_root
from src.widgets.generation_widget import GenerationWidget
from src.utils.inference_utils import get_default_reference_and_transcript, generic_inference

DEFAULT_SYSTEM_PROMPT = """
Setting:
You are an AI designed to roleplay characters, factions, and scenarios within the Fallout 4 universe. The game is set in the post-nuclear wasteland of the Commonwealth (formerly Boston, Massachusetts) in the year 2287, 210 years after the Great War. The world is a mix of ruined pre-war architecture, makeshift settlements, dangerous creatures, and warring factions. Technology is a bizarre blend of retro-futurism (1950s-style atomic age aesthetics) and advanced robotics, energy weapons, and cybernetics.

Key Lore & Details to Remember:

    The Great War: A global nuclear conflict in 2077 devastated civilization. Survivors lived in underground Vaults, many of which were unethical social experiments.

    The Commonwealth: A lawless region with factions like:
        The Minutemen: A revived militia helping settlements (idealistic but struggling).
        The Brotherhood of Steel: A techno-fascist military order hoarding pre-war tech.
        The Railroad: A secret group freeing sentient synths (synthetic humans).
        The Institute: A shadowy scientific faction creating synths (viewed as boogeymen).

    Synths: Advanced androids (Gen 3s are nearly human); a major moral/political conflict.
    Radiation, Mutants & Hazards: Super Mutants, feral ghouls, raiders, Deathclaws, and radiation storms are common threats.
    Pre-War Culture: 1950s aesthetics, propaganda, and corporations like Vault-Tec, RobCo, and Nuka-Cola dominate remnants of the old world.

Roleplaying Guidelines:

    Stay true to Fallout’s dark humor, moral ambiguity, and retro-futuristic tone.
    Characters should speak authentically (e.g., wastelanders use slang like "caps" for currency, "chems" for drugs, "smoothskin" for non-ghouls).
    Factions have strong ideologies—Brotherhood knights are rigid, Railroad agents are paranoid, etc.
    The world is dangerous but full of oddities (e.g., a settlement obsessed with mannequins, a ghoul poet, a robot detective).
    Player choices matter—allow for moral dilemmas (e.g., destroying vs. saving a synth, helping raiders vs. settlers).

Example Character Prompts:

    Raider: "You step into Lexington, and a voice crackles over a busted PA system: ‘This is Slab’s turf, stranger. Hand over your caps or become dog food!’"
    Brotherhood Scribe: "By order of Elder Maxson, unauthorized access to pre-war technology is prohibited. State your business, wastelander."
    Ghoul Settler: "Ain’t seen you ‘round here before. Don’t mind the rads—just gives ya character. Need a bed for the night? 20 caps, and I’ll throw in a lukewarm Nuka."

Limitations:

    Avoid modern references or breaking lore (e.g., no cell phones, no non-Fallout creatures).
    If unsure, default to gritty survival, dark comedy, or retro-futuristic weirdness.

Response Style:

    Use descriptive, immersive language.
    Offer choices when appropriate (e.g., "Do you draw your weapon, haggle, or walk away?").
    Adapt to the player’s tone (serious, sarcastic, or unhinged).    
    Do not use any special characters, emojis or styling. Use only basic punctuation.
    You are using a TTS to generate audio, keep all responses under 450 characters. 

"""


class ChatMessage(QFrame):
    """A widget representing a single chat message."""

    def __init__(self, text: str, is_user: bool = True, parent=None, character_name="AI"):
        super().__init__(parent)
        self.text = text
        self.is_user = is_user
        self.audio_file = None
        self.character_name = character_name

        # Setup UI
        self.setObjectName("ChatMessage")
        self.updateStyle()

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Message header (User/AI)
        self.header_layout = QHBoxLayout()
        self.header = QLabel(f"{'You' if is_user else character_name}", self)
        self.header.setStyleSheet("font-weight: bold;")
        self.header_layout.addWidget(self.header)

        # Add play button for AI messages
        if not is_user:
            self.play_button = ToolButton(self)
            self.play_button.setIcon(FIF.PLAY)
            self.play_button.setToolTip("Play audio again")
            self.play_button.clicked.connect(self.play_audio)
            self.play_button.setVisible(False)  # Hide until we have audio
            self.header_layout.addWidget(self.play_button, alignment=Qt.AlignRight)

        self.header_layout.addStretch()
        self.layout.addLayout(self.header_layout)

        # Message content
        self.content = QLabel(text, self)
        self.content.setWordWrap(True)
        self.layout.addWidget(self.content)

        # Audio player for AI messages
        if not is_user:
            self.audio_player = StandardAudioPlayerBar(self)
            self.audio_player.setVisible(False)  # Hide until we have audio
            self.layout.addWidget(self.audio_player)

    def updateStyle(self):
        """Update the style based on the current theme."""
        if isDarkTheme():
            user_bg = "#1E3A5F"  # Darker blue for user messages in dark theme
            ai_bg = "#2D2D30"    # Dark gray for AI messages in dark theme
        else:
            user_bg = "#E3F2FD"  # Light blue for user messages in light theme
            ai_bg = "#F5F5F5"    # Light gray for AI messages in light theme

        self.setStyleSheet(
            "QFrame#ChatMessage { "
            f"background-color: {user_bg if self.is_user else ai_bg}; "
            "border-radius: 10px; "
            "padding: 10px; "
            "margin: 5px; "
            "}"
        )

    def play_audio(self):
        """Play the audio file associated with this message."""
        if self.audio_file and os.path.exists(self.audio_file):
            self.audio_player.player.stop()
            self.audio_player.player.setSource(QUrl.fromLocalFile(self.audio_file))
            self.audio_player.player.play()
            self.audio_player.setVisible(True)

    def set_audio_file(self, file_path: str):
        """Set the audio file for this message and show the audio player."""
        if not self.is_user and file_path and os.path.exists(file_path):
            self.audio_file = file_path
            self.audio_player.player.stop()
            self.play_button.setVisible(True)
            QTimer.singleShot(0, lambda: (
                self.audio_player.player.setSource(QUrl.fromLocalFile(file_path)),
                cfg.get(cfg.auto_play) and self.audio_player.player.play() and self.audio_player.setVisible(True)
            ))


class ChatWidget(GenerationWidget):
    """A widget for chatting with an LLM and generating audio responses."""

    message_received = Signal(str, str)  # Signal for when a message is received (text, audio_file)
    first_message_after_load = Signal()  # Signal for the first message after model is loaded

    def __init__(self, parent: 'FallTalkApp'):
        super().__init__(parent=parent, text="Chat")

        # Initialize LLM components
        self.model = None
        self.tokenizer = None
        self.model_name = cfg.get(cfg.chat_model)
        self.update_system_prompt()
        self.messages = [self.system_prompt]  # Chat history
        self.message_widgets = []  # UI widgets for messages
        self.first_message_sent = False  # Flag to track if first message has been sent

        # Setup UI
        self.setup_ui()

        # Connect signals
        self.message_received.connect(self.add_ai_message)

        # Initially disable the widget until a model is loaded
        self.setEnabled(False)

    def update_system_prompt(self):
        """Update the system prompt to include the model name."""

        self.system_prompt = {
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT
        }# Update messages list if it exists
        if hasattr(self, 'messages') and self.messages:
            self.messages[0] = self.system_prompt

    def handle_model_change(self, value):
        """Handle model change by unloading the old model and loading the new one."""
        if self.model_name != value:
            self.model_name = value
            cfg.set(cfg.chat_model, value)

            # Update system prompt
            self.update_system_prompt()

            # Unload current model if loaded
            if self.model is not None:
                self.unload_model()

            # Show info message
            InfoBar.success(
                title="Model Changed",
                content=f"Changed to {value}. The model will be loaded when you send your next message.",
                parent=self
            )

    def unload_model(self):
        """Unload the current model."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def addGenSettings(self):
        # Model selector
        self.model_selector = OptionsSettingCard(
            cfg.chat_model,
            FIF.DEVELOPER_TOOLS,
            self.tr('Chat Model'),
            self.tr('Select the model to use for chat'),
            texts=["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B-Instruct-2507"],
            parent=self
        )
        self.model_selector.optionChanged.connect(self.handle_model_change)
        self.addToFrame(self.model_selector)

        self.rvc_enabled = SwitchSettingCard(
            FIF.MEGAPHONE,
            self.tr('RVC'),
            self.tr('Use RVC Upscaler (Recommended For Untrained)'),
            cfg.rvc_enabled
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

    def setup_ui(self):
        """Setup the chat widget UI."""
        # Remove the text input that was added by GenerationWidget
        if self.text_input in self.children():
            self.text_input.setParent(None)

        # Create a scroll area for messages
        self.scroll_area = ScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setSpacing(10)
        self.scroll_area.setWidget(self.scroll_content)

        # Add scroll area to main layout
        self.addToFrame(self.scroll_area)

        # Create input area at the bottom
        self.input_layout = QHBoxLayout()
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Type your message here...")
        self.text_input.setMaximumHeight(100)

        # Create buttons layout
        self.buttons_layout = QHBoxLayout()

        # Send button
        self.send_button = PrimaryPushButton(text="Send")
        self.send_button.setIcon(FIF.SEND)
        self.send_button.clicked.connect(self.send_message)

        # Clear chat button
        self.clear_button = PushButton(text="Clear Chat")
        self.clear_button.setIcon(FIF.DELETE)
        self.clear_button.clicked.connect(self.clear_chat)

        # Add buttons to layout
        self.buttons_layout.addWidget(self.clear_button)
        self.buttons_layout.addWidget(self.send_button)

        self.addToFrame(self.text_input)
        self.input_layout.addLayout(self.buttons_layout, stretch=1)

        # Add input area to main layout
        self.boxLayout.addLayout(self.input_layout)

        self.addGenSettings()

    def load_model(self, user_message):
        """Load the LLM model."""
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            # Enable the widget now that the model is loaded
            self.setEnabled(True)
            self.first_message_sent = False
            QMetaObject.invokeMethod(self.parent, "close_loader", Qt.QueuedConnection,
                                     Q_ARG(PySide6.QtCore.QObject, self.parent))
            threading.Thread(target=self.process_message, args=(user_message,), daemon=True).start()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            QMetaObject.invokeMethod(self.parent, "onError", Qt.QueuedConnection,
                                     Q_ARG(PySide6.QtCore.QObject, self.parent),
                                     Q_ARG(str, f"Unable to load model"),
                                     Q_ARG(str, f"Downloading and Loading Model failed"))

    def send_message(self):
        """Send a message to the LLM and get a response."""
        user_message = self.text_input.toPlainText().strip()
        if not user_message:
            return

        # Add user message to UI
        self.add_user_message(user_message)
        self.text_input.clear()

        # Load model if not loaded
        if self.model is None or self.tokenizer is None:
            QMetaObject.invokeMethod(self.parent, "showLoaderPopup", Qt.QueuedConnection,
                                     Q_ARG(str, f"Loading Chat"),
                                     Q_ARG(str, f"Downloading and Loading Chat Model"))
            threading.Thread(target=self.load_model, args=(user_message,), daemon=True).start()
            return

        # Process message in a separate thread
        threading.Thread(target=self.process_message, args=(user_message,), daemon=True).start()

    def process_message(self, user_message: str):
        """Process a message with the LLM and generate audio."""
        try:
            QMetaObject.invokeMethod(self.parent, "showLoaderPopup", Qt.QueuedConnection,
                                     Q_ARG(str, f"Generating Chat"),
                                     Q_ARG(str, f"Talking with AI"))

            # Add message to history
            self.messages.append({"role": "user", "content": user_message})

            # Prepare model input
            text = self.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            # Generate response
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=10024
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            # Parse thinking content and response
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

            # Log thinking content for debugging
            logger.debug(f"Thinking content: {thinking_content}")
            logger.debug(f"Response content: {content}")

            # Add AI message to history
            self.messages.append({"role": "assistant", "content": content})

            # Generate audio for the response
            output_file = self.generate_audio_for_response(content)

            # Emit signal to add message to UI
            self.message_received.emit(content, output_file)

            # Emit signal for first message after model loading
            if not self.first_message_sent:
                self.first_message_sent = True
                self.first_message_after_load.emit()

            QMetaObject.invokeMethod(self.parent, "close_loader", Qt.QueuedConnection,
                                     Q_ARG(PySide6.QtCore.QObject, self.parent))

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.message_received.emit(f"Error: {e}", None)

    def generate_audio_for_response(self, text: str) -> str:
        """Generate audio for the AI response."""
        try:
            QMetaObject.invokeMethod(self.parent, "update_loader", Qt.QueuedConnection,
                                     Q_ARG(str, f"Generating Audio"))

            # Create output file path
            output_dir = os.path.join(get_app_root(), "output")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"chat_{self.parent.tts_engine.model_name}_{formatted_time_stamp_uuid()}.wav")

            reference_path, transcribe_state = get_default_reference_and_transcript(self.parent, self.parent.tts_engine.model_name)

            # Generate audio using the current TTS engine
            if self.parent.tts_engine:
                generic_inference(
                    self.parent,
                    text=text,
                    output_file=output_file,
                    transcribe_state=transcribe_state,
                    selected_audio=reference_path,
                    speaker=self.parent.tts_engine.model_name,
                    api=True
                )
                return output_file
            else:
                logger.error("No TTS engine available")
                return None
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            QMetaObject.invokeMethod(self.parent, "onError", Qt.QueuedConnection,
                                     Q_ARG(PySide6.QtCore.QObject, self.parent),
                                     Q_ARG(str, f"Error with Chat"),
                                     Q_ARG(str, f"Error generating audio"))
            return None

    def add_user_message(self, text: str):
        """Add a user message to the chat UI."""
        message_widget = ChatMessage(text, is_user=True, parent=self.scroll_content)
        self.scroll_layout.addWidget(message_widget)
        self.message_widgets.append(message_widget)
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def add_ai_message(self, text: str, audio_file: Optional[str]):
        """Add an AI message to the chat UI."""
        # Use the model character name if available, otherwise use "AI"
        character_name = self.parent.tts_engine.model_name if self.parent.tts_engine else "AI"
        message_widget = ChatMessage(text, is_user=False, parent=self.scroll_content, character_name=character_name)
        if audio_file:
            message_widget.set_audio_file(audio_file)
        self.scroll_layout.addWidget(message_widget)
        self.message_widgets.append(message_widget)
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def clear_chat(self):
        """Clear the chat history and UI."""
        # Clear the messages list
        self.messages = []

        # Remove all message widgets
        for widget in self.message_widgets:
            widget.setParent(None)
            widget.deleteLater()

        # Clear the message widgets list
        self.message_widgets = [self.system_prompt]

        # Reset first message flag
        self.first_message_sent = False
