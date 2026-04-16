from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState

from src.ui_imgui.page import Page


class FaqPage(Page):
    page_id = "faq"
    label = "FAQ"
    icon = "fa-circle-question"
    nav_group = 3
    nav_position = "bottom"
    show_audio = False

    def __init__(self, state: AppState):
        self._state = state

    def draw(self):
        from imgui_bundle import imgui
        from src.config.config import cfg

        imgui.text("Frequently Asked Questions")
        imgui.separator()

        if imgui.collapsing_header("What is FallTalk?"):
            imgui.text_wrapped(
                "FallTalk is a text-to-speech tool designed for creating Fallout 4 voice lines. "
                "It supports multiple TTS engines and can generate game-ready audio files with "
                "lip sync (FUZ/LIP/XWM)."
            )

        if imgui.collapsing_header("How do I get started?"):
            imgui.text_wrapped(
                "1. Select a TTS engine from the Characters tab.\n"
                "2. Load a character model.\n"
                "3. Go to Generation, type your text, and click Generate.\n"
                "4. The audio will be saved to your output directory."
            )

        if imgui.collapsing_header("What engines are available?"):
            imgui.text_wrapped(
                "FallTalk supports: XTTSv2, F5, GPT-SoVITS, FishSpeech, StyleTTS2, DIA, "
                "Llasa, Orpheus, Spark, CSM, Higgs, Chatterbox, DMO Speech 2, Vibe, Qwen3 TTS, "
                "and RVC for voice conversion."
            )

        if imgui.collapsing_header("What is RVC?"):
            imgui.text_wrapped(
                "RVC (Retrieval-based Voice Conversion) converts audio from one voice to another. "
                "You can record your voice or use another TTS engine as input, then convert it "
                "to match a Fallout 4 character's voice."
            )

        if imgui.collapsing_header("How do I create FUZ files?"):
            imgui.text_wrapped(
                "Enable 'Create FUZ' in Features settings. Generated audio will automatically "
                "be converted to XWM format and packaged with LIP files into a FUZ file. "
                "Place these in your mod's Sound/Voice directory."
            )

        if imgui.collapsing_header("My audio sounds robotic / unnatural"):
            imgui.text_wrapped(
                "Try these steps:\n"
                "- Adjust the temperature setting (higher = more expressive)\n"
                "- Use a different reference audio\n"
                "- Try a different TTS engine\n"
                "- Enable RVC post-processing\n"
                "- Enable Audio Enhancement"
            )

        if imgui.collapsing_header("Where are my generated files saved?"):
            imgui.text_wrapped(
                f"Files are saved to the output directory configured in Settings > Paths. "
                f"Current: {cfg.get(cfg.output_dir)}"
            )

        if imgui.collapsing_header("Disclaimer"):
            imgui.text_wrapped(
                "FallTalk is a fan-made tool for generating voice lines for Fallout 4 mods. "
                "The developers are not responsible for misuse of the generated audio. "
                "By using this tool, you agree to use it responsibly and in accordance "
                "with applicable laws and the terms of service of the underlying models."
            )
