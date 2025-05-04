import logging
import os
import subprocess
import sys
import uuid

import numpy as np
import soundfile as sf
from src.utils.filesystem_utils import get_app_root
from src.config.config import cfg

logger = logging.getLogger('falltalk')
logger.setLevel(logging.DEBUG)

def extra_audio_from_bsa(item, filename):
    extract_bsa(item)
    extract_fuz(os.path.join(get_app_root(), f"temp/{filename}.fuz"))
    create_xwm(os.path.join(get_app_root(), f"temp/{filename}.xwm"), os.path.join(get_app_root(), f"temp/{filename}.wav"), False)

    if os.path.exists(os.path.join(get_app_root(),f"temp/{filename}.xwm")):
        os.remove(os.path.join(get_app_root(), f"temp/{filename}.xwm"))
    if os.path.exists(os.path.join(get_app_root(),f"temp/{filename}.fuz")):
        os.remove(os.path.join(get_app_root(),f"temp/{filename}.fuz"))
    if os.path.exists(os.path.join(get_app_root(),f"temp/{filename}.lip")):
        os.remove(os.path.join(get_app_root(),f"temp/{filename}.lip"))


def create_fuz_files(fuz_file, xwm_file, lip_file):
    try:
        fuz_path = os.path.join(get_app_root(), 'resource/apps/BmlFuzEncode.exe')
        command = [fuz_path, fuz_file, xwm_file, lip_file]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
        logger.debug(f"fuz file created {fuz_file}")
    except subprocess.CalledProcessError as e:
        logger.exception("Unable to create fuz: %s", e.stderr)


def create_lip_and_fuz(parent, input_file, sr=44100, api=False, existing_lip=None):
    fuz_file = None
    rs_wav = None
    start = None
    try:
        from datetime import datetime
        start = datetime.now()
        xwm_file = input_file.replace(".wav", ".xwm")
        lip_file = existing_lip if existing_lip is not None else input_file.replace(".wav", ".lip")
        fuz_file = input_file.replace(".wav", ".fuz")

        data, samplerate = sf.read(input_file)
        #length_in_ms = (len(data) / samplerate) * 1000

        if samplerate != sr:
            rs_wav = input_file.replace(".wav", "_44100.wav")
            audio_data = load_audio(input_file, sr)
            sf.write(rs_wav, audio_data, sr, subtype='PCM_16')
            logger.debug(f"file resampled {rs_wav}")

        if existing_lip is None:
            create_lip_files(parent, rs_wav if rs_wav is not None else input_file, lip_file)

        create_xwm(rs_wav if rs_wav is not None else input_file, xwm_file)
        create_fuz_files(fuz_file, xwm_file, lip_file)

        if cfg.get(cfg.keep_only_fuz):
            if api:
                os.remove(input_file)

            os.remove(xwm_file)
            os.remove(lip_file)

    except Exception as e:
        logger.exception("Unable to create lip and fuz file")
    finally:
        if rs_wav is not None and os.path.exists(rs_wav):
            os.remove(rs_wav)

    from datetime import datetime
    end = datetime.now()
    return (end - start).total_seconds(), fuz_file


def create_lip_files(parent, input_file, lip_file):
    rs_wav = None
    txt_file = None

    try:
        resp = parent.transcription_engine.transcribe(input_file)
        facefx_path = os.path.join(get_app_root(), "resource/apps/lipgen/FaceFXWrapper.exe")
        facefx_cdf_path = os.path.join(get_app_root(), "resource/apps/lipgen/FonixData.cdf")
        txt_file = input_file.replace(".wav", ".txt")

        if resp['transcript'] is not None and len(resp['transcript']) > 0:
            with open(txt_file, 'w', encoding='utf-8') as file:
                file.write(resp['transcript'])

            rs_wav = input_file.replace(".wav", "_16000.wav")
            rs_data = load_audio(input_file, 16000)
            sf.write(rs_wav, rs_data, 16000, subtype='PCM_16')

            command = [facefx_path, 'Fallout4', 'USEnglish', os.path.abspath(facefx_cdf_path), os.path.abspath(rs_wav), os.path.abspath(lip_file), resp['transcript']]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
            logger.debug(f"lip file created {lip_file}")

        else:
            logger.exception("Unable to create lip files: not transcript generated")

    except subprocess.CalledProcessError as e:
        logger.exception("Unable to create lip files: %s", e.stderr)
    finally:
        if rs_wav is not None and os.path.exists(rs_wav):
            os.remove(rs_wav)

        if txt_file is not None and cfg.get(cfg.keep_only_fuz):
            os.remove(txt_file)


def create_xwm(input, output, encode=True):
    xwma_path = os.path.join(get_app_root(), 'resource/apps/xWMAEncode.exe')
    if encode:
        command = [xwma_path, '-b', '32000', input, output]
    else:
        command = [xwma_path, input, output]
    try:
        # Use subprocess.Popen with creationflags to hide the command window
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
        logger.debug(f"xwm processed {input} {output}")
    except subprocess.CalledProcessError as e:
        logger.exception("Conversion failed: %s", e.stderr)


def extract_fuz(file):
    fuz_path = os.path.join(get_app_root(), 'resource/apps/BmlFuzDecode.exe')
    command = [fuz_path, file]
    try:
        # Use subprocess.Popen with creationflags to hide the command window
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
    except subprocess.CalledProcessError as e:
        logger.exception("Unable to extract fuz: %s", e.stderr)


def extract_bsa(item):
    fallout4_dir = cfg.get(cfg.fallout_4_directory)
    bsa_path = os.path.join(get_app_root(), 'resource/apps/BSABrowser/bsab.exe')
    bsa = f"{fallout4_dir}\\Data\\{item['arcname']}"
    bsa_filter = f"Sound\\Voice\\{item['plugin']}\\{item['folder']}\\{item['filename']}"
    command = [bsa_path, "-e:N", "-f", bsa_filter, bsa, os.path.join(get_app_root(), "temp/")]
    try:
        # Use subprocess.Popen with creationflags to hide the command window
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
    except subprocess.CalledProcessError as e:
        logger.exception("BSA Extraction Failed: %s", e.stderr)


def combine_wav_files(input_files, target_rate=24500, silence_duration=0.5):
    combined_data = None
    for input_file in input_files:
        audio_data = load_audio(input_file, target_rate)

        if combined_data is None:
            combined_data = audio_data
        else:
            # Add silence
            silence_samples = int(silence_duration * target_rate)
            silence = np.zeros(silence_samples, dtype=np.int16)
            combined_data = np.concatenate((combined_data, silence, audio_data))

    unique_id = uuid.uuid4()
    file_name = os.path.join(get_app_root(), f"temp/{unique_id.hex}.wav")
    sf.write(file_name, combined_data, target_rate)
    return file_name


def is_stereo(audio_array):
    return audio_array.ndim == 2


def load_audio(file, sampling_rate, channels=1):
    try:
        # Set the ffmpeg_path variable based on the operating system
        if sys.platform == "win32":
            ffmpeg_path = os.path.join(get_app_root(), os.path.join("ffmpeg.exe"))
        else:
            ffmpeg_path = "ffmpeg"  # Default path for Linux and macOS

        # Initialize the process variable
        process = subprocess.Popen(
            [ffmpeg_path, "-y", "-i", file, "-f", "f32le", "-acodec", "pcm_f32le", "-af", "aresample=resampler=soxr", "-ac", f"{channels}", "-ar", str(sampling_rate), "pipe:1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        out, err = process.communicate()
        if process.returncode != 0:
            print(f"FFmpeg error: {err.decode('utf-8')}")  # Debug statement
            raise RuntimeError(f"FFmpeg error: {err.decode('utf-8')}")
        # print("Audio loaded successfully")  # Debug statement
    except Exception as error:
        print(f"Error loading audio: {error}")  # Debug statement
        raise RuntimeError(f"Failed to load audio: {error}")

    return np.frombuffer(out, np.float32).flatten()