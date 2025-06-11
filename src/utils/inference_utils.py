from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

import logging
import asyncio
import re
from PySide6.QtCore import QMetaObject, Qt, Q_ARG
import PySide6

from src.config.config import cfg
from src.utils.audio_utils import create_lip_and_fuz
from num2words import num2words

logger = logging.getLogger('falltalk')
logger.setLevel(logging.DEBUG)


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


def replace_numbers_with_words(sentence):
    sentence = re.sub(r'(\d+)', r' \1 ', sentence)  # add spaces around numbers

    def replace_with_words(match):
        num = match.group(0)
        try:
            return num2words(num)  # Convert numbers to words
        except:
            return num  # In case num2words fails (unlikely with digits but just to be safe)

    return re.sub(r'\b\d+\b', replace_with_words, sentence)  # Regular expression that matches numbers


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


def xtts_inference(parent: 'FallTalkApp', output_file, text, selected_audio, panel, transcribe_state, speaker=None, api=False):
    generic_inference(parent, output_file, text, selected_audio, panel, api=api)

def dia_inference(parent: 'FallTalkApp', output_file, text, selected_audio, panel, transcribe_state, speaker=None, api=False):
    generic_inference(parent, output_file, text, selected_audio, panel, transcribe_state, api=api)

def orpheus_inference(parent: 'FallTalkApp', output_file, text, selected_audio, panel, transcribe_state, speaker=None, api=False):
    generic_inference(parent, output_file, text, selected_audio, panel, transcribe_state, api=api, speaker=speaker)

def llasa_inference(parent: 'FallTalkApp', output_file, text, selected_audio, panel, transcribe_state, speaker=None, api=False):
    generic_inference(parent, output_file, text, selected_audio, panel, transcribe_state, api=api, speaker=speaker)

def fish_inference(parent: 'FallTalkApp', output_file, text, selected_audio, panel, transcribe_state, speaker=None, api=False):
    generic_inference(parent, output_file, text, selected_audio, panel, transcribe_state, api=api)

def f5_inference(parent: 'FallTalkApp', output_file, text, selected_audio, panel, start_time, end_time, transcribe_state, speaker=None, api=False):
    generic_inference(parent, output_file, text, selected_audio, panel, transcribe_state, start_time, end_time, api=api)

def gpt_sovits_inference(parent: 'FallTalkApp', output_file, text, selected_audio, panel, transcribe_state, speaker=None, api=False):
    generic_inference(parent, output_file, text, selected_audio, panel, transcribe_state, api=api, speaker=speaker)

def styletts2_inference(parent: 'FallTalkApp', output_file, text, selected_audio, panel, transcribe_state, speaker=None, api=False):
    generic_inference(parent, output_file, text, selected_audio, panel, api=api)

def spark_inference(parent: 'FallTalkApp', output_file, text, selected_audio, panel, transcribe_state, speaker=None, api=False):
    generic_inference(parent, output_file, text, selected_audio, panel, transcribe_state, api=api, speaker=speaker)

def csm_inference(parent: 'FallTalkApp', output_file, text, selected_audio, panel, transcribe_state, speaker=None, api=False):
    generic_inference(parent, output_file, text, selected_audio, panel, transcribe_state, api=api, speaker=speaker)
