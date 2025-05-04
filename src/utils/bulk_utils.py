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

from src.config.config import cfg
from src.utils.audio_utils import create_lip_and_fuz
from src.utils.file_utils import get_bulk_folder
from src.utils.filesystem_utils import get_app_root
from src.utils.model_utils import get_character_model, get_trained_character
from src.utils.huggingface_utils import download_models, download_rvc_models
from src.utils.inference_utils import (
    do_transcribe, rvc_inference, xtts_inference, gpt_sovits_inference, styletts2_inference
)

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

        tts_engine.run_rvc(output_file)

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
    wav_files = glob.glob(os.path.join(directory, '**', '*.wav'), recursive=include_subdir)
    fuz_files = glob.glob(os.path.join(directory, '**', '*.fuz'), recursive=include_subdir)
    xwm_files = glob.glob(os.path.join(directory, '**', '*.xwm'), recursive=include_subdir)
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

    if is_trained:
        # if parent.tts_engine.engine_name != 'RVC':
        #     engine_changed = True
        #     load_rvc(parent, True)

        if not os.path.exists(os.path.join('models', model['name'], 'RVC')):
            download_rvc_models(model['name'], model['RVC'])

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
            tts_engine.setup(model['name'], has_rvc, not is_trained)
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


def bulk_inference(parent):
    if parent.tts_engine.engine_name != 'RVC':
        model = parent.bulk_generate_widget.bulk_csv_widget.bulk_table.model()
        datas = model.getData()
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
                reference_path = None

                model = get_character_model(character, parent.models, parent.custom_models)
                is_trained, has_rvc = get_trained_character(model, parent.tts_engine.engine_name)

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

                if reference_voice is not None and os.path.exists(reference_voice):
                    reference_path = reference_voice
                elif reference_voice is not None:
                    reference_path = get_reference(parent, character, reference_voice)

                if last_character is None or last_character != character:
                    if is_trained and not os.path.exists(os.path.join('models', character, parent.tts_engine.engine_name)):
                        download_models(parent, character, model[parent.tts_engine.engine_name], model['RVC'] if has_rvc else None, True)
                    elif has_rvc and not os.path.exists(os.path.join('models', character, 'RVC')):
                        download_rvc_models(character, model['RVC'])

                    if is_trained and character != parent.tts_engine.model_name:
                        parent.tts_engine.setup(character, has_rvc, False)

                    elif not is_trained and character != parent.tts_engine.model_name:
                        parent.tts_engine.setup(character, has_rvc, True)

                is_wav = os.path.exists(text_or_file)

                if has_rvc and is_wav:
                    data, samplerate = sf.read(text_or_file)
                    sf.write(output_file, data, samplerate)
                    rvc_inference(parent, output_file, None, True)
                elif not is_wav and parent.tts_engine.engine_name == 'GPT_SoVITS':
                    transcript = None
                    if not is_trained and reference_path is not None:
                        resp = do_transcribe(parent, os.path.abspath(reference_path), None, api=True)
                        transcript = resp['transcript'] if resp is not None else None
                    gpt_sovits_inference(parent, output_file, text_or_file, os.path.abspath(reference_path) if parent.tts_engine.is_base else [os.path.abspath(reference_path)], None, transcript, api=True)
                elif not is_wav and parent.tts_engine.engine_name == 'XTTSv2':
                    xtts_inference(parent, output_file, text_or_file, reference_path, None, api=True)
                elif not is_wav and parent.tts_engine.engine_name == 'StyleTTS2':
                    styletts2_inference(parent, output_file, text_or_file, reference_path, None, api=True)

                if cfg.get(cfg.xwm_enabled):
                    create_lip_and_fuz(parent, output_file, 44100, True)

            except Exception as e:
                logger.exception(f"bulk_inference failed for row {data}")

            count += 1
            end = datetime.now()
            time_total += (end - start).total_seconds()
            update_progress(parent, total, time_total, count)

    QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))