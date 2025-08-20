from __future__ import annotations

import json
import os
import random
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

import logging
import asyncio
import re
from PySide6.QtCore import QMetaObject, Qt, Q_ARG

import PySide6

from src.config.config import cfg
from src.utils.audio_utils import create_lip_and_fuz, extra_audio_from_bsa, combine_wav_files
from src.utils.filesystem_utils import get_app_root
from src.enums.engine_type import EngineType

from num2words import num2words

logger = logging.getLogger('falltalk')
logger.setLevel(logging.DEBUG)



def get_default_reference(parent, character_name):
    """Get a default reference for a character from default_references.json and characters.json

    Args:
        parent: The parent application instance
        character_name: The name of the character

    Returns:
        tuple: (reference_path, transcript, filename) or (None, None, None) if no reference found
"""


    currentEgine = parent.tts_engine.engine_type
    max_reference_length = currentEgine.max_reference_length
    min_reference_length = currentEgine.min_reference_length


    # First check if we have default references for this character
    if hasattr(parent, 'default_references') and parent.default_references and character_name in parent.default_references:
        # Get all default references for this character
        default_refs = parent.default_references[character_name]
        if default_refs:
            # Create a list to hold references that meet the duration requirement
            valid_references = []
            combined_transcript = ""
            total_duration = 0
            
            # Keep track of selected references to avoid duplicates
            selected_refs = []
            
            # Keep selecting references until we meet the minimum duration requirement
            while len(valid_references) < len(default_refs) and len(valid_references) < 5:  # Limit to 5 to prevent infinite loops
                # Filter out already selected references
                available_refs = [ref for ref in default_refs if ref not in selected_refs]
                
                # If no more references are available, break the loop
                if not available_refs:
                    break

                random.shuffle(available_refs)
                    
                # Select a random default reference from available ones
                random_ref = random.choice(available_refs)
                
                # Add to selected references to avoid picking it again
                selected_refs.append(random_ref)

                # Get the filename and transcript from default_references.json
                wav_filename = random_ref.get('filename', '')
                transcript = random_ref.get('transcript', '')
                ref_duration = random_ref.get('duration', 0)

                # Convert WAV filename to FUZ filename to match with characters.json
                # Example: "00112D18_1.wav" -> "00112d18_1.fuz"
                fuz_filename = wav_filename.replace('.wav', '.fuz').lower()

                # Now find the matching entry in characters.json to get BSA info
                if character_name in parent.characters_data:
                    character = parent.characters_data[character_name]
                    if character.get('voicefiles'):
                        # Find the matching voice file in characters.json
                        matching_voice_file = None
                        for voice_file in character['voicefiles']:
                            if voice_file.get('filename', '').lower() == fuz_filename:
                                matching_voice_file = voice_file
                                break

                        if matching_voice_file:
                            # Create a temporary WAV file for the reference
                            temp_dir = os.path.join(get_app_root(), "temp")
                            os.makedirs(temp_dir, exist_ok=True)
                            temp_file = os.path.join(temp_dir, wav_filename)

                            # Check if the file already exists in the temp directory
                            if not os.path.exists(temp_file):
                                # Try to extract from BSA if it's a game file
                                matching_voice_file['folder'] = character_name
                                try:
                                    extra_audio_from_bsa(matching_voice_file, fuz_filename.replace('.fuz', ''))
                                except Exception as e:
                                    logger.warning(f"Could not extract audio from BSA: {e}")
                                    # Continue to try other methods if this fails

                            # If the file exists now, add it to our valid references
                            if os.path.exists(temp_file):
                                # Get actual duration by reading the file once
                                actual_duration = 0
                                try:
                                    import soundfile as sf
                                    audio_data, sample_rate = sf.read(temp_file)
                                    actual_duration = len(audio_data) / sample_rate
                                except Exception as e:
                                    # If we can't read the file, use the stored duration
                                    actual_duration = ref_duration
                                
                                valid_references.append(temp_file)
                                combined_transcript += " " + transcript if combined_transcript else transcript
                                
                                # Add to total duration using actual duration
                                total_duration += actual_duration
                                
                                # Check if we have enough duration
                                if total_duration >= min_reference_length:
                                    break

            # If we have valid references, process them
            if valid_references:
                # If we have multiple references, combine them
                if len(valid_references) > 1:
                    combined_file = combine_wav_files(valid_references, target_rate=44100)
                    return combined_file, combined_transcript.strip(), os.path.basename(combined_file), None
                else:
                    # Just return the single valid reference
                    return valid_references[0], combined_transcript.strip(), os.path.basename(valid_references[0]), None

    # Fallback to using characters.json directly if default_references.json didn't work
    if character_name not in parent.characters_data:
        logger.warning(f"Character {character_name} not found in characters data")
        return None, None, None, None

    character = parent.characters_data[character_name]
    if not character.get('voicefiles'):
        logger.warning(f"No voice files found for character {character_name}")
        return None, None, None, None

    # Sort voice files by dialogue length (longer is better for reference)
    voice_files = []
    for voice_file in character['voicefiles']:
        if voice_file.get('filename') and voice_file.get('dialogue'):
            voice_files.append((voice_file, len(voice_file['dialogue'])))

    if not voice_files:
        logger.warning(f"No valid voice files found for character {character_name}")
        return None, None, None, None

    # Sort by dialogue length and get the top one
    voice_files.sort(key=lambda x: x[1], reverse=True)
    voice_file, _ = voice_files[0]

    # Get the filename and dialogue
    filename = voice_file['filename']
    transcript = voice_file.get('dialogue', '')

    # Create a temporary WAV file for the reference
    temp_dir = os.path.join(get_app_root(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    wav_filename = filename.replace('.fuz', '.wav')
    temp_file = os.path.join(temp_dir, wav_filename)

    # Check if the file already exists in the temp directory
    if not os.path.exists(temp_file):
        # Try to extract from BSA if it's a game file
        voice_file['folder'] = character_name
        try:
            extra_audio_from_bsa(voice_file, filename.replace('.fuz', ''))
        except Exception as e:
            logger.warning(f"Could not extract audio from BSA: {e}")
            return None, None, None, None

    # If the file exists now, return it
    if os.path.exists(temp_file):
        return temp_file, transcript, wav_filename, None

    return None, None, None, None


def get_default_reference_and_transcript(parent, character_name):
    """Get a default reference audio and transcript for a character.

    Args:
        parent: The parent application instance
        character_name: The name of the character

    Returns:
        tuple: (reference_path, transcribe_state) where reference_path is the path to the reference audio file
               and transcribe_state is a dictionary containing the transcript
    """
    # Get default reference from default_references.json and characters.json
    reference_path, transcript, filename, duration = get_default_reference(parent, character_name)

    # If we have a transcript, create the transcribe state
    transcribe_state = None
    if transcript:
        transcribe_state = {'transcript': transcript}

    return reference_path, transcribe_state

def do_transcribe_before_gen(parent: 'FallTalkApp', selected_audio):
    try:
        resp = parent.transcription_engine.transcribe(selected_audio)
        logger.debug("transcription complete {}",resp)
        QMetaObject.invokeMethod(parent, "after_transcribe_gen", Qt.QueuedConnection,
                                 Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, json.dumps(resp)))
        return resp
    except Exception as e:
        logger.exception("Transcription failed")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent),
                                 Q_ARG(str, "Unable to Transcribe Audio"), Q_ARG(str,
                                                                                 "An Error Occured while attempting to transcribe audio. Please check your logs and report the issue if needed"))

def do_transcribe(parent: 'FallTalkApp', selected_audio, widget, api=False):
    try:
        resp = parent.transcription_engine.transcribe(selected_audio)
        if not api:
            widget.transcribe_state = resp
            QMetaObject.invokeMethod(parent, "after_transcribe", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(PySide6.QtCore.QObject, widget))
        return resp
    except Exception as e:
        logger.exception("Transcription failed")
        if not api:
            QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Transcribe Audio"), Q_ARG(str, "An Error Occured while attempting to transcribe audio. Please check your logs and report the issue if needed"))


def ensure_sentence_punctuation(text):
    """
    Ensures that the text ends with standard punctuation (., !, ?).
    If it ends with non-standard punctuation, removes it and adds a period.
    If it doesn't end with any punctuation, adds a period.
    """
    if not text:
        return None

    # Define the standard punctuation marks
    standard_punctuation = ".!?"
    # All punctuation characters
    import string
    all_punctuation = string.punctuation

    text = text.strip()

    if text:
        # Check if the last character is a punctuation mark
        if text[-1] in all_punctuation:
            # If it's not a standard punctuation mark, replace it with a period
            if text[-1] not in standard_punctuation:
                text = text[:-1] + "."
        else:
            # If it's not a punctuation mark at all, add a period
            text += "."

    return text


def replace_numbers_with_words(sentence):
    """
    Replaces numbers in text with their word equivalents.
    """
    if sentence:
        sentence = re.sub(r'(\d+)', r' \1 ', sentence)  # add spaces around numbers

        def replace_with_words(match):
            num = match.group(0)
            try:
                return num2words(num)  # Convert numbers to words
            except:
                return num  # In case num2words fails (unlikely with digits but just to be safe)

        return re.sub(r'\b\d+\b', replace_with_words, sentence)  # Regular expression that matches numbers
    else:
        return None


def remove_emojis(text):
    # Regex pattern to match most emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F700-\U0001F77F"  # Alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r'', text)

def normalize_whitespace(text):
    """
    Normalizes whitespace by removing extra spaces and newlines.
    """
    if text:
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Trim leading and trailing whitespace
        text = text.strip()
        return text
    return None


def convert_to_lowercase(text):
    """
    Converts text to lowercase.
    """
    if text:
        return text.lower()
    return None


def fix_dot_letters(text):
    """
    Converts patterns like "J.R.R." to "J R R" to improve initialisms and names.
    """
    if text:
        # Find patterns like X.Y.Z. where X, Y, Z are single letters
        return re.sub(r'(\b[A-Za-z]\.)+', lambda m: m.group(0).replace('.', ' ').strip(), text)
    return None


def remove_inline_references(text):
    """
    Removes numbers after sentence-ending punctuation (e.g., .188 or ."3).
    """
    if text:
        # Find patterns like .[number] or ."[number]
        return re.sub(r'([\.\!\?][\"\']?)(\d+)', r'\1', text)
    return None


def split_text(text, max_length=300, min_length=30):
    """
    Splits text into chunks of maximum length, trying to split at natural boundaries.

    Args:
        text (str): The text to split
        max_length (int): Maximum length of each chunk
        min_length (int): Minimum length of each chunk

    Returns:
        list: List of text chunks
    """
    if not text:
        return []

    # If text is already short enough, return it as is
    if len(text) <= max_length:
        return [text]

    # Define split points in order of preference
    split_points = ['. ', '; ', ': ', '- ', ', ']

    chunks = []
    current_chunk = ""

    # Split text into sentences first
    sentences = []
    current_sentence = ""

    for char in text:
        current_sentence += char
        if char in '.!?' and len(current_sentence) > 0:
            sentences.append(current_sentence)
            current_sentence = ""

    # Add any remaining text as a sentence
    if current_sentence:
        sentences.append(current_sentence)

    # Process sentences into chunks
    for sentence in sentences:
        # If adding this sentence would exceed max_length
        if len(current_chunk) + len(sentence) > max_length:
            # If current_chunk is too small, we need to split the sentence
            if len(current_chunk) < min_length:
                # Try to find a good split point in the sentence
                split_found = False
                for split_char in split_points:
                    if split_char in sentence:
                        parts = sentence.split(split_char, 1)
                        if len(current_chunk) + len(parts[0] + split_char) >= min_length:
                            current_chunk += parts[0] + split_char
                            chunks.append(current_chunk)
                            current_chunk = parts[1]
                            split_found = True
                            break

                # If no good split point, just add as much as we can
                if not split_found:
                    available_space = max_length - len(current_chunk)
                    if available_space > 0:
                        current_chunk += sentence[:available_space]
                        chunks.append(current_chunk)
                        current_chunk = sentence[available_space:]
                    else:
                        chunks.append(current_chunk)
                        current_chunk = sentence
            else:
                # Current chunk is big enough, add it and start a new one
                chunks.append(current_chunk)
                current_chunk = sentence
        else:
            # Add sentence to current chunk
            current_chunk += sentence

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    # Handle any chunks that are still too large
    result = []
    for chunk in chunks:
        if len(chunk) > max_length:
            # Split by character count if still too large
            result.extend([chunk[i:i+max_length] for i in range(0, len(chunk), max_length)])
        else:
            result.append(chunk)

    return result


def preprocess_text(text):
    """
    Applies text preprocessing functions based on configuration:
    1. Lowercase conversion (optional)
    2. Whitespace normalization (optional)
    3. Dot-letter fix (optional)
    4. Inline reference number removal (optional)
    5. Ensures text ends with punctuation
    6. Replaces numbers with words

    Returns the processed text.
    """
    if not text:
        return None

    # Apply optional preprocessing based on configuration
    if cfg.get(cfg.lowercase_conversion):
        text = convert_to_lowercase(text)

    if cfg.get(cfg.whitespace_normalization):
        text = normalize_whitespace(text)

    if cfg.get(cfg.dot_letter_fix):
        text = fix_dot_letters(text)

    if cfg.get(cfg.inline_reference_removal):
        text = remove_inline_references(text)

    # Always apply these preprocessing steps
    text = remove_emojis(text)
    text = ensure_sentence_punctuation(text)
    text = replace_numbers_with_words(text)

    return text


def get_eleven_labs_voices():
    from elevenlabs.client import ElevenLabs
    key = cfg.get(cfg.rvc_eleven_labs_key)
    key = None if key == '' else key
    client = ElevenLabs(
        api_key=key
    )
    response = client.voices.get_all(show_legacy=True)
    tts_voice_list = response.voices
    return sorted([f"{v.name}" for v in tts_voice_list])


def get_edge_tts_voices():
    import edge_tts
    tts_voice_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())
    return [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list if v['ShortName'].startswith('en-')]


def eleven_labs_inference(parent: 'FallTalkApp', text, output_file, voice, panel, api=False):
    from elevenlabs.core import ApiError
    try:
        key = cfg.get(cfg.rvc_eleven_labs_key)
        key = None if key == '' else key
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(
            api_key=key
        )
        audio = client.generate(
            text=text,
            voice=voice,
            model="eleven_multilingual_v2"
        )
        from elevenlabs import save
        save(audio, output_file)
        rvc_inference(parent, output_file, panel, api)
    except ApiError as api_error:
        if not api:
            QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Call Eleven Labs"), Q_ARG(str, str(api_error.body)))
    except Exception as e:
        logger.exception("Eleven Labs inference failed")
        if not api:
            QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Call Eleven Labs"), Q_ARG(str, "An Error Occured while attempting to generate audio. Please check your logs and report the issue if needed"))


def edge_tts_inference(parent: 'FallTalkApp', text, output_file, voice, panel, api=False):
    try:
        import edge_tts

        asyncio.run(
            edge_tts.Communicate(
                text, "-".join(voice.split("-")[:-1])
            ).save(output_file)
        )
        rvc_inference(parent, output_file, panel, api)
    except Exception as e:
        logger.exception("Edge TTS inference failed")
        if not api:
            QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Generate Audio"), Q_ARG(str, "An Error Occured while attempting to generate audio. Please check your logs and report the issue if needed"))


def generic_inference(parent: 'FallTalkApp', output_file, text, selected_audio=None, panel=None, transcribe_state=None, start_time=None, end_time=None, api=False, speaker=None):
    try:
        # Common parameters for all engines
        kwargs = {
            'text': text,
            'voice': selected_audio,
            'language': "en",
            'output_file': output_file
        }

        # Add optional parameters if they exist
        if transcribe_state is not None:
            if isinstance(transcribe_state, dict) and parent.tts_engine.engine_type != EngineType.F5:
                kwargs['transcript'] = transcribe_state['transcript']
            else:
                kwargs['transcript'] = transcribe_state
        if start_time is not None:
            kwargs['start_time'] = start_time
        if end_time is not None:
            kwargs['end_time'] = end_time
        if speaker is not None:
            kwargs['speaker'] = speaker

        # Generate audio with the engine
        parent.tts_engine.generate_audio(**kwargs)

        if not api:
            QMetaObject.invokeMethod(parent, "updateMediaplayer", Qt.QueuedConnection, 
                                   Q_ARG(PySide6.QtCore.QObject, panel), 
                                   Q_ARG(str, output_file))
            if cfg.get(cfg.xwm_enabled):
                create_lip_and_fuz(parent, output_file)
            QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, 
                                   Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"{parent.tts_engine.engine_name} inference failed")
        if not api:
            QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, 
                                   Q_ARG(PySide6.QtCore.QObject, parent), 
                                   Q_ARG(str, "Unable to Generate Audio"), 
                                   Q_ARG(str, "An Error Occurred while attempting to generate audio. Please check your logs and report the issue if needed"))


def rvc_inference(parent: 'FallTalkApp', input_file, panel, api=False):
    try:
        parent.tts_engine.run_rvc_file(input_file)
        if not api:
            QMetaObject.invokeMethod(parent, "updateMediaplayer", Qt.QueuedConnection, 
                                   Q_ARG(PySide6.QtCore.QObject, panel), 
                                   Q_ARG(str, input_file))
            if cfg.get(cfg.xwm_enabled):
                create_lip_and_fuz(parent, input_file)
            QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, 
                                   Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception("RVC inference failed")
        if not api:
            QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, 
                                   Q_ARG(PySide6.QtCore.QObject, parent), 
                                   Q_ARG(str, "Unable to Generate Audio"), 
                                   Q_ARG(str, "An Error Occurred while attempting to generate audio. Please check your logs and report the issue if needed"))
