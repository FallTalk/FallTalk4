"""
Consolidated engine help — replaces all 20+ src/help/*.py files.
Single draw_engine_help(state) dispatches to per-engine help text.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState

from src.enums.engine_type import EngineType

# -- Help text constants (sourced from src/help/<engine>_help.py) --

XTTS_HELP = """\
XTTSv2 (eXtended Text-To-Speech)

About:
- XTTS is a versatile and powerful text-to-speech model.
- It excels at producing highly natural speech with accurate pronunciation and intonation.
- The model is designed to handle a wide range of languages and speaking styles.

Tips:
- For best results, provide well-structured text with appropriate punctuation.
- The model performs exceptionally well with both short phrases and longer narrative content.
- When generating multilingual content, make sure to provide clear language indicators if needed.
- XTTS is particularly good at maintaining consistent voice characteristics throughout longer passages.
- Using RVC upscaler can further enhance the quality and naturalness of the generated speech.
"""

F5_HELP = """\
F5 Text-to-Speech

About:
- If the generated speech is too fast, adjust the speed factor.

Tips:
- For best results, provide clear and well-punctuated text.
- The model works best with sentences of moderate length. Very long sentences may be broken up during processing.
- If you need to generate speech with specific emotional qualities, try adjusting the temperature and other parameters.
"""

GPT_SOVITS_HELP = """\
GPT-SoVITS Text-to-Speech

Settings Guide:
- Temperature: Randomness in the generation.
  Low (e.g., 0.2) = very predictable, safe outputs.
  High (e.g., 1.2) = more variety, but you might get odd or jumbled speech.
- Top P: Only pick words from the smallest group whose total probability reaches p (e.g., 0.9 means the top 90% most likely words). Keeps things coherent by ignoring the long tail of rare options.
- Top K: Restricts each choice to the k most likely words (e.g., k=50). Smaller k makes speech more focused; larger k adds risk of weirdness.
"""

FISH_HELP = """\
FishSpeech Text-to-Speech

About:
- Fish is a versatile text-to-speech model designed for generating natural-sounding speech.
- It offers good performance across a variety of speaking styles and content types.
- The model is optimized for efficiency while maintaining high-quality output.

Tips:
- For optimal results, provide text with proper punctuation and structure.
- The model handles both short phrases and longer paragraphs effectively.
- If generating dialogue, consider adding appropriate punctuation to indicate pauses and intonation changes.
- Experiment with different parameter settings to find the voice characteristics that best suit your needs.
"""

STYLETTS2_HELP = """\
StyleTTS2 Text-to-Speech

About:
- StyleTTS2 is an advanced text-to-speech model that focuses on stylistic control and expressiveness.
- It allows for fine-grained control over speaking style, emotion, and prosody.
- The model is particularly effective at capturing and reproducing specific voice characteristics.

Tips:
- For optimal results, provide well-structured text with appropriate punctuation.
- The model excels at maintaining consistent speaking style throughout longer passages.
- When generating dialogue, consider adding context or emotion indicators to guide the model's expression.
- Experiment with different parameter settings to achieve the desired speaking style and emotional tone.
- StyleTTS2 works particularly well for content that requires distinctive character voices or emotional expression.
"""

DIA_HELP = """\
DIA (Dynamic Intonation Adjustment)

About:
- DIA is a text-to-speech model that focuses on natural intonation patterns.
- It provides high-quality voice synthesis with dynamic pitch and rhythm control.
- The model is particularly good at capturing emotional nuances in speech.

Tips:
- Temperature: Controls randomness in generation. Higher values create more varied speech, while lower values produce more consistent results.
- Speed: Adjusts the pace of the generated speech.
- Top P and Top K: Fine-tune the creativity and coherence of the generated speech.
- Slice Mode: For longer texts, choose how to break down the input for processing.
"""

LLASA_HELP = """\
LLASA (Large Language and Speech Assistant)

About:
- LLASA combines language understanding with speech synthesis.
- It leverages advanced AI techniques to generate highly natural and contextually appropriate speech.
- The model is particularly effective at maintaining consistent speaking style and intonation patterns.

Tips:
- For best results, provide clear and well-structured text input.
- The model performs well with both short commands and longer narrative content.
- If generating dialogue, consider including speaker indicators or context cues.
- Experiment with different temperature settings to find the right balance between consistency and expressiveness.
- Using RVC upscaler can significantly enhance the quality and naturalness of the generated speech.
"""

ORPHEUS_HELP = """\
Orpheus Text-to-Speech

About:
- Orpheus is a sophisticated text-to-speech model named after the legendary musician of Greek mythology.
- It excels at producing highly musical and expressive speech with natural prosody.
- The model is particularly effective for generating speech with emotional depth and nuanced intonation.

Tips:
- For optimal results, provide text with appropriate punctuation to guide the model's pacing and intonation.
- The model performs exceptionally well with dialogue and narrative content that requires emotional expression.
- Consider using commas and other punctuation marks strategically to create natural pauses.
- Experiment with different temperature settings: higher values create more varied and expressive speech.
- For longer texts, breaking them into meaningful segments can help maintain consistent quality.
"""

SPARK_HELP = """\
Spark Text-to-Speech

About:
- Spark is a powerful text-to-speech model designed for generating highly natural and expressive speech.
- It uses advanced neural network architectures to produce speech with realistic intonation and rhythm.
- The model is capable of handling a wide range of speaking styles and content types.

Tips:
- For best results, provide well-structured text with appropriate punctuation.
- The model performs particularly well with conversational content and narrative text.
- When generating longer passages, consider breaking them into logical segments.
- Experiment with different temperature settings for the right balance between consistency and expressiveness.
- Using RVC upscaler can significantly enhance the clarity and naturalness of the generated speech.
"""

CSM_HELP = """\
CSM Text-to-Speech

About:
- CSM is a compact text-to-speech model optimized for efficient speech generation.
- It provides good quality output while being resource-efficient.

Tips:
- Adjust temperature to control the randomness of generated speech.
- Lower temperature values produce more predictable output.
"""

HIGGS_HELP = """\
Higgs Audio Text-to-Speech

About:
- Higgs Audio is a high-quality text-to-speech model that can generate natural-sounding speech.
- It uses reference audio to match the voice characteristics and speaking style.

Tips:
- For best results, provide clear and well-punctuated text.
- The model works best with sentences of moderate length.
- If you need to generate speech with specific emotional qualities, try adjusting the temperature.
- The RAS (Repetition Avoidance Sampling) parameters help prevent repetitive patterns in the generated audio.
"""

CHATTERBOX_HELP = """\
Chatterbox Text-to-Speech

General Use:
- The default settings (exaggeration=0.5, cfg_weight=0.5) work well for most prompts.
- If the reference speaker has a fast speaking style, lowering cfg_weight to around 0.3 can improve pacing.

Expressive or Dramatic Speech:
- Try lower cfg_weight values (e.g. ~0.3) and increase exaggeration to around 0.7 or higher.
- Higher exaggeration tends to speed up speech; reducing cfg_weight helps compensate with slower, more deliberate pacing.

Tips:
- For best results, provide well-structured text with appropriate punctuation.
- The model performs particularly well with conversational content and narrative text.
- Experiment with different temperature settings for the right balance between consistency and expressiveness.
- Using RVC upscaler can significantly enhance the clarity and naturalness of the generated speech.
"""

DMO_HELP = """\
DMO Speech 2 Text-to-Speech

About:
- DMO Speech 2 is a text-to-speech model with teacher-student architecture.
- It provides controllable speech generation with adjustable parameters.

Tips:
- Adjust temperature to control randomness in generation.
- Teacher steps and stopping time control the teacher model behavior.
"""

VIBE_HELP = """\
Vibe Text-to-Speech

About:
- Vibe is a text-to-speech model with configurable sampling and CFG guidance.
- Available in 1.5B (7GB VRAM) and 7B (24GB VRAM) model sizes.

Tips:
- Use Sampling allows more randomness but is less stable.
- CFG Scale controls classifier-free guidance strength.
- Inference Steps: more steps increase generation time but may improve quality.
"""

QWEN_HELP = """\
Qwen3 TTS

About:
- Qwen3 TTS is a multilingual text-to-speech model.
- Available in 1.7B-Base and 0.6B-Base model sizes.

Tips:
- Use the Instruct field to guide the speaker's style (e.g., "Very happy.", "Whispered.").
- Select the appropriate language for best results.
"""

RVC_HELP = """\
RVC (Retrieval-based Voice Conversion)

About:
- RVC is a powerful tool for transforming audio from one voice to another.
- It allows you to convert any voice input into the voice of a character from Fallout 4 or other custom voices.
- This is particularly useful for creating custom voice lines for mods or for voice acting with character voices.

RVC Modes:
- Microphone: Record your voice directly and convert it to a character voice in real-time.
- File: Convert an existing audio file (WAV or MP3) to a character voice.
- Edge TTS: Use Microsoft's free text-to-speech service as a base voice, then convert it.
- ElevenLabs: Use ElevenLabs' high-quality TTS as a base voice (requires API key).

Tips:
- Ensure your input audio is clear and has minimal background noise.
- Adjust the pitch setting if the voice sounds unnatural (especially for cross-gender conversion).
- "Index Influence Ratio" controls how much voice characteristics are applied. Higher = more detail but may introduce artifacts.
- "Filter Radius" >= 3 can reduce breathing sounds in the output.
- "Autotune" can improve singing voice conversions.
- For longer audio files, enable "Split Audio" to process in chunks.
- "Create FUZ" will generate game-ready files with lip synchronization.
"""


def draw_engine_help(state: AppState):
    """Render engine-specific help content."""
    from imgui_bundle import imgui

    if state.engine_type is None:
        imgui.text("Load an engine to see help.")
        return

    match state.engine_type:
        case EngineType.F5:
            imgui.text_wrapped(F5_HELP)
        case EngineType.GPT_SOVITS:
            imgui.text_wrapped(GPT_SOVITS_HELP)
        case EngineType.FISH_SPEECH:
            imgui.text_wrapped(FISH_HELP)
        case EngineType.STYLE_TTS2:
            imgui.text_wrapped(STYLETTS2_HELP)
        case EngineType.DIA:
            imgui.text_wrapped(DIA_HELP)
        case EngineType.LLASA:
            imgui.text_wrapped(LLASA_HELP)
        case EngineType.ORPHEUS:
            imgui.text_wrapped(ORPHEUS_HELP)
        case EngineType.SPARK:
            imgui.text_wrapped(SPARK_HELP)
        case EngineType.CSM:
            imgui.text_wrapped(CSM_HELP)
        case EngineType.HIGGS:
            imgui.text_wrapped(HIGGS_HELP)
        case EngineType.CHATTERBOX:
            imgui.text_wrapped(CHATTERBOX_HELP)
        case EngineType.DMOSPEECH2:
            imgui.text_wrapped(DMO_HELP)
        case EngineType.VIBE:
            imgui.text_wrapped(VIBE_HELP)
        case EngineType.QWEN3_TTS:
            imgui.text_wrapped(QWEN_HELP)
        case EngineType.RVC:
            imgui.text_wrapped(RVC_HELP)
        case _:
            imgui.text(f"No help available for {state.engine_type}")
