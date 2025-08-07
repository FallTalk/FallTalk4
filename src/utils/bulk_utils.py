from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

import logging

import os
import glob
import time
import shutil
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from PySide6.QtCore import QMetaObject, Qt, Q_ARG
import PySide6
import random

from src.config.config import cfg
from src.utils.audio_utils import create_lip_and_fuz, extra_audio_from_bsa
from src.utils.file_utils import get_bulk_folder, clean_tmp_folder
from src.utils.filesystem_utils import get_app_root
from src.utils.model_utils import get_character_model, get_trained_character
from src.utils.huggingface_utils import download_models, download_rvc_models
from src.utils.inference_utils import (
    do_transcribe, rvc_inference, xtts_inference, gpt_sovits_inference, styletts2_inference, dia_inference,
    f5_inference, fish_inference, orpheus_inference, llasa_inference, csm_inference, spark_inference, higgs_inference,
    chatterbox_inference, dmo_speech2_inference, preprocess_text
)
from src.enums.engine_type import EngineType

logger = logging.getLogger('falltalk')
logger.setLevel(logging.DEBUG)


def update_progress(parent, total, time_total, count):
    additional_data_points_needed = total - count
    avg_duration = time_total / count
    estimated_duration = (avg_duration * additional_data_points_needed)
    td = timedelta(seconds=estimated_duration)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    QMetaObject.invokeMethod(parent, "update_loader", Qt.QueuedConnection, Q_ARG(str, f"Completed: {count}/{total}. Estimated Duration: {hours:02}:{minutes:02}:{seconds:02}"))


def get_reference(parent, character, reference):
    character_model = parent.characters_data[character] if character in parent.characters_data else None
    if character_model is not None:
        for voice_file in character_model['voicefiles']:
            if reference in voice_file['filename']:
                filename = f"{voice_file['filename']}".replace('.fuz', '')
                if not os.path.exists(os.path.join(get_app_root(), f"temp/{filename}.wav")):
                    voice_file['folder'] = character
                    from src.utils.audio_utils import extra_audio_from_bsa
                    extra_audio_from_bsa(voice_file, filename)
                return os.path.join(get_app_root(), f"temp/{filename}.wav")


def process_xwm_file(xwm_file, cfg, files, use_existing_lip=False):
    wav_file = xwm_file.replace(".xwm", ".wav")
    from src.utils.audio_utils import create_xwm
    create_xwm(xwm_file, wav_file, encode=False)

    if cfg.get(cfg.replace_existing):
        if os.path.exists(xwm_file):
            os.remove(xwm_file)

    if not use_existing_lip and os.path.exists(xwm_file.replace(".xwm", ".lip")):
        os.remove(xwm_file.replace(".xwm", ".lip"))

    files.add(wav_file)


def process_fuz_file(fuz_file, cfg, files, use_existing_lip=False):
    from src.utils.audio_utils import extract_fuz, create_xwm
    extract_fuz(fuz_file)
    xwm_file = fuz_file.replace(".fuz", ".xwm")
    wav_file = fuz_file.replace(".fuz", ".wav")
    create_xwm(xwm_file, wav_file, encode=False)
    files.add(wav_file)

    if cfg.get(cfg.replace_existing):
        if os.path.exists(fuz_file):
            os.remove(fuz_file)

    if os.path.exists(xwm_file):
        os.remove(xwm_file)
    if not use_existing_lip and os.path.exists(xwm_file.replace(".xwm", ".lip")):
        os.remove(xwm_file.replace(".xwm", ".lip"))


def process_rvc_file(tts_engine, wav_file, replace, output_folder, parent, directory, use_existing_lip=False):
    start = datetime.now()
    try:
        existing_lip = None
        if not replace:
            relative_path = os.path.relpath(wav_file, directory)
            if os.path.dirname(relative_path) != '':
                sub_dir = os.path.split(relative_path)[0]
                output_file = os.path.join(output_folder, sub_dir, os.path.basename(wav_file))
                os.makedirs(os.path.join(output_folder, sub_dir), exist_ok=True)
            else:
                output_file = os.path.join(output_folder, os.path.basename(wav_file))

            shutil.copy(wav_file, output_file)

            if use_existing_lip and os.path.exists(wav_file.replace(".wav", ".lip")):
                existing_lip = os.path.join(output_folder, os.path.basename(wav_file.replace(".wav", ".lip")))
                shutil.copy(wav_file.replace(".wav", ".lip"), existing_lip)

        else:
            output_file = wav_file
            if use_existing_lip and os.path.exists(wav_file.replace(".wav", ".lip")):
                existing_lip = wav_file.replace(".wav", ".lip")

        tts_engine.run_rvc_file(output_file)

        if cfg.get(cfg.xwm_enabled):
            create_lip_and_fuz(parent, output_file, 44100, True, existing_lip)
    except Exception as e:
        logger.exception("RVC inference failed")

    end = datetime.now()
    return (end - start).total_seconds()


def bulk_fuz(parent, directory, include_subdir, threads=1, use_existing_lip=True):
    wav_files = glob.glob(os.path.join(directory, '**', '*.wav'), recursive=include_subdir)
    xwm_files = glob.glob(os.path.join(directory, '**', '*.xwm'), recursive=include_subdir)
    count = 0
    time_total = 0
    files = set()
    files.update(wav_files)
    from src.utils.model_utils import load_whisper
    load_whisper(parent)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []

        for xwm_file in xwm_files:
            futures.append(executor.submit(process_xwm_file, xwm_file, cfg, files, False))

        for future in as_completed(futures):
            future.result()

    total = len(files)

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for idx, wav_file in enumerate(files):

            existing_lip = None
            if use_existing_lip and os.path.exists(wav_file.replace(".wav", ".lip")):
                existing_lip = wav_file.replace(".wav", ".lip")

            future = executor.submit(create_lip_and_fuz, parent, wav_file, 44100, True, existing_lip)
            futures.append(future)

        for future in as_completed(futures):
            time_taken, fuz_file = future.result()
            time_total += time_taken
            count += 1
            update_progress(parent, total, time_total, count)

    QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))


def bulk_rvc_inference(parent, directory, model, include_subdir, replace, threads=1, use_existing_lip=True):
    start_time = time.time()

    if include_subdir:
        pattern_base = os.path.join(directory, '**')
        recursive = True
    else:
        pattern_base = directory
        recursive = False

    wav_files = glob.glob(os.path.join(pattern_base, '*.wav'), recursive=recursive)
    fuz_files = glob.glob(os.path.join(pattern_base, '*.fuz'), recursive=recursive)
    xwm_files = glob.glob(os.path.join(pattern_base, '*.xwm'), recursive=recursive)

    count = 0
    time_total = 0
    output_folder = None
    # model = get_character_model(character, parent.models, parent.custom_models)
    is_trained, has_rvc = get_trained_character(model, 'RVC')
    engine_changed = False

    files = set()

    files.update(wav_files)

    from src.utils.model_utils import load_whisper
    load_whisper(parent)

    tts_engines = []

    if output_folder is None and not replace:
        output_folder = get_bulk_folder(model['display_name'])
        os.makedirs(output_folder, exist_ok=True)

    if is_trained and has_rvc:
        # if parent.tts_engine.engine_name != 'RVC':
        #     engine_changed = True
        #     load_rvc(parent, True)

        if not os.path.exists(os.path.join('models', model['name'], 'RVC')):
            download_rvc_models(parent, model['name'], model['RVC'])

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []

            for xwm_file in xwm_files:
                futures.append(executor.submit(process_xwm_file, xwm_file, cfg, files, use_existing_lip))

            for fuz_file in fuz_files:
                futures.append(executor.submit(process_fuz_file, fuz_file, cfg, files, use_existing_lip))

            for future in as_completed(futures):
                future.result()

        total = len(files)

        QMetaObject.invokeMethod(parent, "update_loader", Qt.QueuedConnection, Q_ARG(str, f"Setting Up RVC Engines: {threads}"))
        for i in range(threads):
            from src.tts_engines.rvc_engine import RVC_Engine
            tts_engine = RVC_Engine()
            tts_engine.setup(model['name'], has_rvc, not is_trained, "1")
            tts_engine.preload_rvc_params()
            tts_engines.append(tts_engine)

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for idx, wav_file in enumerate(files):
                tts_engine = tts_engines[idx % threads]
                future = executor.submit(process_rvc_file, tts_engine, wav_file, replace, output_folder, parent, directory, use_existing_lip)
                futures.append(future)

            for future in as_completed(futures):
                time_taken = future.result()
                time_total += time_taken
                count += 1
                update_progress(parent, total, time_total, count)

    end_time = time.time()
    elapsed_time = end_time - start_time
    td = timedelta(seconds=elapsed_time)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Bulk Duration: {hours:02}:{minutes:02}:{seconds:02}")

    for engine in tts_engines:
        engine.clean()
        del engine

    if engine_changed:
        QMetaObject.invokeMethod(parent, "afterRVC", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    else:
        QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))




def process_inference_data(parent, data, output_file, character, model, is_trained, has_rvc, reference_voice, text_or_file, api=True, last_character=None):
    """Helper function to process inference data for both bulk and ez voice creator"""
    import os

    try:
        if last_character is None or last_character != character:
            clean_tmp_folder()
            if is_trained and not os.path.exists(os.path.join('models', character, parent.tts_engine.engine_name)):
                download_models(parent, character, model[parent.tts_engine.engine_name], model['RVC'] if has_rvc else None, True)
            elif has_rvc and not os.path.exists(os.path.join('models', character, 'RVC')):
                download_rvc_models(parent, character, model['RVC'])

            if parent.tts_engine.is_shared:
                if character not in parent.tts_engine.characters:
                    if is_trained and character != parent.tts_engine.model_name:
                        engine_model = model[parent.tts_engine.engine_name]
                        parent.tts_engine.setup(character, has_rvc, False, engine_model.get('engine_version', "1"), engine_model.get('is_shared', False), engine_model.get('shared_model_name', None),  engine_model.get('characters', None))
                    elif not is_trained and character != parent.tts_engine.model_name:
                        parent.tts_engine.setup(character, has_rvc, True)
            elif is_trained and character != parent.tts_engine.model_name:
                engine_model = model[parent.tts_engine.engine_name]
                parent.tts_engine.setup(character, has_rvc, False, engine_model.get('engine_version', "1"), engine_model.get('is_shared', False), engine_model.get('shared_model_name', None),  engine_model.get('characters', None))
            elif not is_trained and character != parent.tts_engine.model_name:
                parent.tts_engine.setup(character, has_rvc, True)

        is_wav = os.path.exists(text_or_file)

        # Get reference audio
        reference_path = None
        transcript = None
        transcribe_state = None

        if reference_voice is not None and os.path.exists(reference_voice):
            reference_path = reference_voice
        elif reference_voice is not None and reference_voice != "":
            reference_path = get_reference(parent, character, reference_voice)
        else:
            # If no reference file specified or it doesn't exist, and model is not trained
            engine_type = next((e for e in EngineType if e.value == parent.tts_engine.engine_name), None)
            if not is_trained or engine_type.needs_reference_when_trained:
                # Get default reference from characters.json
                reference_path, transcript, _ = get_default_reference(parent, character)

                # If we have a transcript, create the transcribe state
                if transcript:
                    transcribe_state = {'transcript': transcript}

                # If no reference found, log warning and return
                if reference_path is None:
                    logger.warning(f"No reference files found for character {character}")
                    return

        if has_rvc and is_wav:
            data, samplerate = sf.read(text_or_file)
            sf.write(output_file, data, samplerate)
            rvc_inference(parent, output_file, None, api)
        elif not is_wav:
            # Apply text preprocessing (ensure punctuation and replace numbers with words)
            text_or_file = preprocess_text(text_or_file)

            engine_type = next((e for e in EngineType if e.value == parent.tts_engine.engine_name), None)
            if engine_type is None:
                raise ValueError(f"Invalid engine type: {parent.tts_engine.engine_name}")

            # Only transcribe if we don't already have a transcript from default references
            if not transcript or not transcribe_state:
                transcript = None
                transcribe_state = None
                if engine_type.needs_transcription and reference_path is not None:
                    resp = do_transcribe(parent, os.path.abspath(reference_path), None, api=True)
                    transcribe_state = resp if resp is not None else None
                    transcript = transcribe_state['transcript'] if transcribe_state is not None else None

            if engine_type == EngineType.GPT_SOVITS:
                gpt_sovits_inference(parent, output_file, text_or_file, reference_path, None, transcript, api)
            elif engine_type == EngineType.XTTS_V2:
                xtts_inference(parent, output_file, text_or_file, reference_path, None, api)
            elif engine_type == EngineType.STYLE_TTS2:
                styletts2_inference(parent, output_file, text_or_file, reference_path, None, api)
            elif engine_type == EngineType.DIA:
                dia_inference(parent, output_file, text_or_file, reference_path, None, transcript, api=api, speaker=character)
            elif engine_type == EngineType.F5:
                f5_inference(parent, output_file, text_or_file, reference_path, None, None, None, transcribe_state=transcribe_state, api=api)
            elif engine_type == EngineType.FISH_SPEECH:
                fish_inference(parent, output_file, text_or_file, reference_path, None, transcript, api)
            elif engine_type == EngineType.ORPHEUS:
                orpheus_inference(parent, output_file, text_or_file, reference_path, None, transcript, api=api, speaker=character)
            elif engine_type == EngineType.LLASA:
                llasa_inference(parent, output_file, text_or_file, reference_path, None, transcript, api=api, speaker=character)
            elif engine_type == EngineType.CSM:
                csm_inference(parent, output_file, text_or_file, reference_path, None, transcript, api=api, speaker=character)
            elif engine_type == EngineType.SPARK:
                spark_inference(parent, output_file, text_or_file, reference_path, None, transcript, api=api, speaker=character)
            elif engine_type == EngineType.HIGGS:
                higgs_inference(parent, output_file, text_or_file, reference_path, None, transcript, api=api, speaker=character)
            elif engine_type == EngineType.CHATTERBOX:
                chatterbox_inference(parent, output_file, text_or_file, reference_path, None, transcript, api=api, speaker=character)
            elif engine_type == EngineType.DMOSPEECH2:
                dmo_speech2_inference(parent, output_file, text_or_file, reference_path, None, transcript, api=api, speaker=character)

        if cfg.get(cfg.xwm_enabled):
            create_lip_and_fuz(parent, output_file, 44100, True)
    except Exception as e:
        logger.exception(f"Inference failed for row {data}")
        raise


def bulk_inference(parent):
    if parent.tts_engine.engine_name != EngineType.RVC.value:
        model = parent.bulk_generate_widget.bulk_csv_widget.bulk_table.model()
        datas = model.getData()

        # Sort data by character (index 1) to process all entries for the same character together
        datas.sort(key=lambda x: x[1] if x[1] is not None else "")

        count = 0
        total = len(datas)
        time_total = 0

        generic_output_folder = get_bulk_folder(parent.tts_engine.engine_name)
        last_character = None

        for data in datas:
            start = datetime.now()
            try:
                file_name = data[0]
                character = data[1]
                text_or_file = data[2]
                reference_voice = data[3]
                output_folder = data[4]

                model = get_character_model(character, parent.models, parent.custom_models)
                is_trained, has_rvc = get_trained_character(model, parent.tts_engine.engine_name)
                engine_type = next((e for e in EngineType if e.value == parent.tts_engine.engine_name), None)
                if engine_type is None:
                    raise ValueError(f"Invalid engine type: {parent.tts_engine.engine_name}")

                if file_name is not None and file_name != "":
                    if ".wav" not in file_name:
                        file_name = file_name + ".wav"
                else:
                    import uuid
                    unique_id = uuid.uuid4()
                    file_name = f"{unique_id.hex[:10]}.wav"

                if output_folder is None or output_folder == "":
                    output_folder = generic_output_folder

                output_file = f"{output_folder}/{file_name}"

                try:
                    os.makedirs(output_folder, exist_ok=True)
                except OSError as e:
                    pass

                process_inference_data(parent, data, output_file, character, model, is_trained, has_rvc, reference_voice, text_or_file, last_character=last_character)
                last_character = character

            except Exception as e:
                logger.exception(f"bulk_inference failed for row {data}")

            count += 1
            end = datetime.now()
            time_total += (end - start).total_seconds()
            update_progress(parent, total, time_total, count)

    QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))


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


def ez_voice_creator_inference(parent):
    if parent.tts_engine.engine_name != EngineType.RVC.value:
        model = parent.ez_voice_creator_widget.dialogue_table.model()
        table = parent.ez_voice_creator_widget.dialogue_table
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Only get data from visible (not filtered out) rows
        datas = []
        for row in range(model.rowCount()):
            if not table.isRowHidden(row):
                datas.append(model.getData()[row])

        # Sort data by voice_type (index 2) to process all entries for the same character together
        datas.sort(key=lambda x: x[2] if x[2] is not None else "")

        count = 0
        total = len(datas)
        time_total = 0

        last_character = None

        for data in datas:
            start = datetime.now()
            try:
                file_name = data[0]  # FILE_NAME
                text = data[1]       # RESPONSE TEXT
                voice_type = data[2] # VOICE TYPE
                full_path = data[3]  # FULLPATH
                reference_voice = data[4]  # REFERENCE FILE
                plugin_name = data[5]  # PLUGIN
                plugin_name = f"{plugin_name}_{timestamp}"

                # Get character data
                character = parent.characters_data.get(voice_type)
                if not character:
                    continue

                model = get_character_model(character['name'], parent.models, parent.custom_models)
                is_trained, has_rvc = get_trained_character(model, parent.tts_engine.engine_name)
                engine_type = next((e for e in EngineType if e.value == parent.tts_engine.engine_name), None)
                if engine_type is None:
                    raise ValueError(f"Invalid engine type: {parent.tts_engine.engine_name}")

                # Construct the output path with plugin name
                # Remove the plugin name from the full path if it exists
                path_parts = full_path.split(os.sep)
                if len(path_parts) > 1 and path_parts[0] == 'Data':
                    path_parts = path_parts[1:]  # Remove 'Data' from the path

                output_path = os.path.join(get_app_root(), 'bulk_outputs/', plugin_name, 'Data', *path_parts)
                output_file = output_path.replace('.fuz', '.wav')

                # Create the directory structure if it doesn't exist
                output_dir = os.path.dirname(output_file)
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except OSError as e:
                    logger.exception(f"Failed to create directory {output_dir}: {e}")
                    continue

                process_inference_data(parent, data, output_file, character['name'], model, is_trained, has_rvc, reference_voice, text, last_character=last_character)
                last_character = character['name']

            except Exception as e:
                logger.exception(f"ez_voice_creator_inference failed for row {data}")

            count += 1
            end = datetime.now()
            time_total += (end - start).total_seconds()
            update_progress(parent, total, time_total, count)

    QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
