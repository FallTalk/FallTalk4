from __future__ import annotations

import json
import os
import random
from typing import TYPE_CHECKING

from enums.engine_type import EngineType

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

import logging
import asyncio
import re
from PySide6.QtCore import QMetaObject, Qt, Q_ARG, QMetaType, QObject

import PySide6

from src.config.config import cfg
from src.utils.audio_utils import create_lip_and_fuz, extra_audio_from_bsa
from src.utils.filesystem_utils import get_app_root

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
    # First check if we have default references for this character
    if hasattr(parent, 'default_references') and parent.default_references and character_name in parent.default_references:
        # Get a random default reference for this character
        default_refs = parent.default_references[character_name]
        if default_refs:
            # Select a random default reference
            random_ref = random.choice(default_refs)

            # Get the filename and transcript from default_references.json
            wav_filename = random_ref.get('filename', '')
            transcript = random_ref.get('transcript', '')

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

                        # If the file exists now, return it
                        if os.path.exists(temp_file):
                            return temp_file, transcript, wav_filename

    # Fallback to using characters.json directly if default_references.json didn't work
    if character_name not in parent.characters_data:
        logger.warning(f"Character {character_name} not found in characters data")
        return None, None, None

    character = parent.characters_data[character_name]
    if not character.get('voicefiles'):
        logger.warning(f"No voice files found for character {character_name}")
        return None, None, None

    # Sort voice files by dialogue length (longer is better for reference)
    voice_files = []
    for voice_file in character['voicefiles']:
        if voice_file.get('filename') and voice_file.get('dialogue'):
            voice_files.append((voice_file, len(voice_file['dialogue'])))

    if not voice_files:
        logger.warning(f"No valid voice files found for character {character_name}")
        return None, None, None

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
            return None, None, None

    # If the file exists now, return it
    if os.path.exists(temp_file):
        return temp_file, transcript, wav_filename

    return None, None, None


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
    reference_path, transcript, filename = get_default_reference(parent, character_name)

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


def preprocess_text(text):
    """
    Applies both text preprocessing functions:
    1. Ensures text ends with punctuation
    2. Replaces numbers with words

    Returns the processed text.
    """
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
            if isinstance(transcribe_state, dict) and parent.tts_engine.engin_type != EngineType.F5:
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
