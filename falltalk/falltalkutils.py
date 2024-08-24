import asyncio
import glob
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import PySide6
import huggingface_hub
import librosa
import numpy as np
import requests
import soundfile as sf
from PySide6.QtCore import QMetaObject, Qt, Q_ARG
from num2words import num2words

import config
from falltalk.config import cfg, REPO

logger = logging.getLogger('falltalk')
logger.setLevel(logging.DEBUG)


def extra_audio_from_bsa(item, filename):
    extract_bsa(item)
    extract_fuz(os.path.abspath(f"temp/{filename}.fuz"))
    create_xwm(os.path.abspath(f"temp/{filename}.xwm"), os.path.abspath(f"temp/{filename}.wav"), False)

    if os.path.exists(f"temp/{filename}.xwm"):
        os.remove(f"temp/{filename}.xwm")
    if os.path.exists(f"temp/{filename}.fuz"):
        os.remove(f"temp/{filename}.fuz")
    if os.path.exists(f"temp/{filename}.lip"):
        os.remove(f"temp/{filename}.lip")


def create_fuz_files(fuz_file, xwm_file, lip_file):
    try:
        fuz_path = './resource/apps/BmlFuzEncode.exe'
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
    start = datetime.now()
    try:
        xwm_file = input_file.replace(".wav", ".xwm")
        lip_file = existing_lip if existing_lip is not None else input_file.replace(".wav", ".lip")
        fuz_file = input_file.replace(".wav", ".fuz")

        data, samplerate = sf.read(input_file)
        length_in_ms = (len(data) / samplerate) * 1000
        if samplerate != sr:
            rs_wav = input_file.replace(".wav", "_44100.wav")
            audio_data = librosa.resample(data, orig_sr=samplerate, target_sr=sr)
            sf.write(rs_wav, audio_data, sr)
            logger.debug(f"file resampled {rs_wav}")
        elif not api:
            rs_wav = input_file.replace(".wav", "_44100.wav")
            sf.write(rs_wav, data, sr)
            logger.debug(f"file copy {rs_wav}")

        if existing_lip is None and length_in_ms > 500:
            create_lip_files(parent, rs_wav if rs_wav is not None else input_file, lip_file)

        create_xwm(rs_wav if rs_wav is not None else input_file, xwm_file)
        create_fuz_files(fuz_file, xwm_file, lip_file)

        if cfg.get(cfg.keep_only_fuz):
            if api:
                os.remove(input_file)

            os.remove(xwm_file)
            os.remove(lip_file)

        if rs_wav is not None and os.path.exists(rs_wav):
            os.remove(rs_wav)


    except Exception as e:
        logger.exception("Unable to create lip and fuz file")
    end = datetime.now()
    return (end - start).total_seconds(), fuz_file


def create_lip_files(parent, input_file, lip_file):
    try:
        resp = parent.transcription_engine.transcribe(input_file)
        facefx_path = "./resource/apps/lipgen/FaceFXWrapper.exe"
        facefx_cdf_path = "./resource/apps/lipgen/FonixData.cdf"
        txt_file = input_file.replace(".wav", ".txt")

        if resp['transcript'] is not None and len(resp['transcript']) > 0:
            with open(txt_file, 'w', encoding='utf-8') as file:
                file.write(resp['transcript'])

            command = [facefx_path, 'Fallout4', 'USEnglish', os.path.abspath(facefx_cdf_path), os.path.abspath(input_file), os.path.abspath(lip_file), resp['transcript']]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
            logger.debug(f"lip file created {lip_file}")

            if cfg.get(cfg.keep_only_fuz):
                os.remove(txt_file)
        else:
            logger.exception("Unable to create lip files: not transcript generated")

    except subprocess.CalledProcessError as e:
        logger.exception("Unable to create lip files: %s", e.stderr)


def create_xwm(input, output, encode=True):
    xwma_path = './resource/apps/xWMAEncode.exe'
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
    fuz_path = './resource/apps/BmlFuzDecode.exe'
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
    bsa_path = './resource/apps/BSABrowser/bsab.exe'
    bsa = f"{fallout4_dir}\\Data\\{item['arcname']}"
    bsa_filter = f"Sound\\Voice\\{item['plugin']}\\{item['folder']}\\{item['filename']}"
    command = [bsa_path, "-e:N", "-f", bsa_filter, bsa, os.path.abspath("temp/")]
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
    file_name = f"temp/{unique_id.hex}.wav"
    sf.write(file_name, combined_data, target_rate)
    return file_name


def load_audio(file, sampling_rate):
    if ".wav" in file:
        audio_data, sr = librosa.load(file, sr=sampling_rate)

        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        return audio_data
    else:
        try:
            # Set the ffmpeg_path variable based on the operating system
            if sys.platform == "win32":
                ffmpeg_path = os.path.abspath(os.path.join("ffmpeg.exe"))
            else:
                ffmpeg_path = "ffmpeg"  # Default path for Linux and macOS

            print(f"f{ffmpeg_path}")
            # Initialize the process variable
            process = subprocess.Popen(
                [ffmpeg_path, "-y", "-i", file, "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sampling_rate), "pipe:1"],
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


def clean_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.exception("Failed to clean folder")


def clean_path(path_str):
    if platform.system() == 'Windows':
        path_str = path_str.replace('/', '\\')
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ").strip("\u202a")


def do_transcribe(parent, selected_audio, widget, api=False):
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


def load_model(parent, character=None, rvc=None, display_name=None, base_model=False):
    print(f"load_model {parent} {character}")
    try:
        if not character.startswith("custom_"):
            if rvc is not None:
                download_rvc_models(character, rvc)
                downloadRVC(parent)

        if parent.tts_engine is not None:
            if character is not None:
                parent.reference_time_label.setText("00:00")
                parent.character_label.setText(f"{display_name}")
            else:
                parent.reference_time_label.setText("00:00")
                parent.character_label.setText(f"Base Model")

            parent.tts_engine.setup(character, rvc is not None, base_model)
            QMetaObject.invokeMethod(parent, "afterModelLoader", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception("Unable to load model")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Model"), Q_ARG(str, "An Error Occured while attempting to load the model. Please check your logs and report the issue if needed"))


def download_rvc_models(character, rvc):
    if rvc:
        download_model_from_hub(character, rvc)
        rvc['type'] = 'index'
        download_model_from_hub(character, rvc)


def download_models(parent, character, model, rvc, api=False):
    try:
        download_model_from_hub(character, model)
        download_rvc_models(character, rvc)
        if model['engine'] == "GPT_SoVITS":
            model['type'] = "ckpt"
            download_model_from_hub(character, model)

        if not api:
            QMetaObject.invokeMethod(parent, "afterModelDownload", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception("Unable to download model")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Model"), Q_ARG(str, "An Error Occured while downloading the model from huggingface. Please check your logs and report the issue if needed"))


def download_model_from_hub(character, model):
    if model is not None:
        os.makedirs(os.path.join("models", character, model['engine']), exist_ok=True)
        filename = os.path.join(character, model['engine'], f"{character}_v{model['version']}.{model['type']}")
        files_to_delete = glob.glob(os.path.join("models", character, model['engine'], f"{character}*.{model['type']}"))
        for file_path in files_to_delete:
            if file_path != os.path.join("models", str(filename)):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.exception("Unable to delete old model")

        huggingface_hub.hf_hub_download(REPO, f"models/{filename.replace(os.sep, '/')}", local_dir=os.path.abspath(f"./"))


def load_music_gen(parent, api=False):
    try:
        if parent.music_engine is None:
            from tts_engines.musicgen_engine import MusicGenEngine
            # from tts_engines.audioldm2_engine import Audioldm2Engine
            # parent.music_engine = MusicGenEngine(parent, "cvssp/audioldm2-music")
            parent.music_engine = MusicGenEngine(parent)
            print("music engine Loaded")
        if not api:
            QMetaObject.invokeMethod(parent, "after_music_gen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Music Engine"), Q_ARG(str, "An Error Occured while loading the music engine. Please check your logs and report the issue if needed"))


def load_fx_gen(parent, api=False):
    try:
        if parent.sound_fx_engine is None:
            from tts_engines.audiogen_engine import AudioGenEngine
            parent.sound_fx_engine = AudioGenEngine(parent)
            # from tts_engines.audioldm2_engine import Audioldm2Engine
            # parent.sound_fx_engine = Audioldm2Engine(parent, "cvssp/audioldm2-large")
            print("fx engine Loaded")
        if not api:
            QMetaObject.invokeMethod(parent, "after_fx_gen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load FX Engine"), Q_ARG(str, "An Error Occured while loading the fx engine. Please check your logs and report the issue if needed"))


def load_upscaler(parent, api=False):
    try:
        if parent.upscale_engine is None:
            from tts_engines.upscale_engine import UpscaleEngine
            parent.upscale_engine = UpscaleEngine(parent)
            print("Upscaler Loaded")
        if not api:
            QMetaObject.invokeMethod(parent, "after_upscale", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Upscaler"), Q_ARG(str, "An Error Occured while loading the upscaler. Please check your logs and report the issue if needed"))


def download_file_from_web(url, local_path):
    if os.path.exists(local_path):
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Download the file
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path, 'wb') as file:
            file.write(response.content)
    else:
        logger.error(f"Failed to download file. Status code: {response.status_code}")


def load_xtts(parent):
    try:
        downloadXTTS(parent)
        downloadRVC(parent)
        from tts_engines.xtts_engine import XTTS_Engine
        parent.tts_engine = XTTS_Engine()
        print("XTTS Loaded")
        load_whisper(parent)
        QMetaObject.invokeMethod(parent, "afterXtts", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
        QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Engine"), Q_ARG(str, "An Error Occured while loading the engine. Please check your logs and report the issue if needed"))


def load_whisper(parent, attempt=0):
    try:
        if parent.transcription_engine is None:
            from tts_engines.whisper_engine import Whisper_Engine
            parent.transcription_engine = Whisper_Engine()
            print("WhisperX Loaded")
    except Exception as e:
        logger.exception(f"Error: {e}")
        if attempt == 0:
            from huggingface_hub.constants import HF_HUB_CACHE
            cache_dir = HF_HUB_CACHE
            if os.path.exists(HF_HUB_CACHE):
                shutil.rmtree(cache_dir)
                print(f"Attempting to delete HuggingFace cache directory: {cache_dir}")
            load_whisper(parent, 1)
        else:
            QMetaObject.invokeMethod(parent, "onWarn", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Whispper Engine"), Q_ARG(str, "An Error Occured while loading whisper engine. Transcription will not work for untrained models. Please delete C:\\Users\\USERNAME\\.cache\\huggingface\\hub"))



def load_voicecraft(parent):
    try:
        downloadVoiceCraft(parent)
        downloadRVC(parent)
        from tts_engines.voicecraft_engine import VoiceCraft_Engine
        from tts_engines.whisper_engine import Whisper_Engine
        parent.tts_engine = VoiceCraft_Engine()
        print("VoiceCraft Loaded")
        load_whisper(parent)
        QMetaObject.invokeMethod(parent, "afterVoiceCraft", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
        QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Engine"), Q_ARG(str, "An Error Occured while loading the engine. Please check your logs and report the issue if needed"))


def load_gpt_sovits(parent):
    try:
        downloadGPTSoVITS(parent)
        downloadRVC(parent)
        from tts_engines.gpt_sovits_engine import GPT_SoVITS_Engine
        from tts_engines.whisper_engine import Whisper_Engine
        parent.tts_engine = GPT_SoVITS_Engine()
        print("GPT_SoVITS Loaded")
        load_whisper(parent)
        QMetaObject.invokeMethod(parent, "afterGPT_SoVITS", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
        QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Engine"), Q_ARG(str, "An Error Occured while loading the engine. Please check your logs and report the issue if needed"))


def load_rvc(parent, api=False):
    try:
        downloadRVC(parent)
        from tts_engines.rvc_engine import RVC_Engine
        parent.tts_engine = RVC_Engine()
        print("RVC Loaded")
        load_whisper(parent)
        if not api:
            QMetaObject.invokeMethod(parent, "afterRVC", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
            QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Engine"), Q_ARG(str, "An Error Occured while loading the engine. Please check your logs and report the issue if needed"))


def load_style_tts2(parent):
    try:
        downloadStyleTTS2(parent)
        downloadRVC(parent)
        print("StyleTTS2 downloaded")
        from tts_engines.style_tts_engine import StyleTTS2_Engine
        parent.tts_engine = StyleTTS2_Engine()
        print("StyleTTS2 Loaded")
        load_whisper(parent)
        QMetaObject.invokeMethod(parent, "afterStyleTTS2", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
        QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Engine"), Q_ARG(str, "An Error Occured while loading the engine. Please check your logs and report the issue if needed"))


def downloadXTTS(parent):
    try:
        os.makedirs("models/XTTSv2", exist_ok=True)
        huggingface_hub.hf_hub_download(REPO, "models/XTTSv2/model.pth", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/XTTSv2/config.json", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/XTTSv2/vocab.json", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/XTTSv2/speakers_xtts.pth", local_dir=os.path.abspath(f"./"))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


def downloadVoiceCraft(parent):
    try:
        os.makedirs("models/VoiceCraft", exist_ok=True)
        huggingface_hub.hf_hub_download(REPO, "models/VoiceCraft/encodec_4cb2048_giga.th", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/VoiceCraft/config.json", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/VoiceCraft/830M_TTSEnhanced.pth", local_dir=os.path.abspath(f"./"))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


def downloadRVC(parent):
    try:
        os.makedirs("models/RVC", exist_ok=True)
        huggingface_hub.hf_hub_download(REPO, "models/rvc/rmvpe.onnx", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/rvc/rmvpe.pt", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/rvc/hubert_base.pt", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/rvc/fcpe.pt", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/rvc/contentvec_base.pt", local_dir=os.path.abspath(f"./"))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


def downloadFish(parent):
    try:
        os.makedirs("models/fish", exist_ok=True)
        huggingface_hub.hf_hub_download("fishaudio/fish-speech-1.2-sft", "firefly-gan-vq-fsq-4x1024-42hz-generator.pth", local_dir=os.path.abspath(f"models/fish"))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


def downloadGPTSoVITS(parent):
    try:
        os.makedirs("models/GPT_SoVITS", exist_ok=True)
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/v2/s2G2333k.pth", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/v2/s2D2333k.pth", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/v2/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/chinese-roberta-wwm-ext-large/pytorch_model.bin", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/chinese-roberta-wwm-ext-large/tokenizer.json", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/chinese-roberta-wwm-ext-large/config.json", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/chinese-hubert-base/pytorch_model.bin", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/chinese-hubert-base/preprocessor_config.json", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/GPT_SoVITS/chinese-hubert-base/config.json", local_dir=os.path.abspath(f"./"))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


def downloadStyleTTS2(parent):
    try:
        os.makedirs("models/StyleTTS2", exist_ok=True)
        huggingface_hub.hf_hub_download(REPO, "models/StyleTTS2/ASR/config.yml", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/StyleTTS2/ASR/epoch_00080.pth", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/StyleTTS2/JDC/bst.t7", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/StyleTTS2/PLBERT/config.yml", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/StyleTTS2/PLBERT/step_1000000.t7", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/StyleTTS2/Models/Vokan/epoch_2nd_00012.pth", local_dir=os.path.abspath(f"./"))
        huggingface_hub.hf_hub_download(REPO, "models/StyleTTS2/Models/Vokan/config.yml", local_dir=os.path.abspath(f"./"))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


def downloadBaseModels(parent):
    try:
        downloadVoiceCraft(parent)
        downloadXTTS(parent)
        downloadRVC(parent)
        downloadGPTSoVITS(parent)
        downloadStyleTTS2(parent)
        QMetaObject.invokeMethod(parent, "afterDownload", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Download Models"), Q_ARG(str, "An Error Occured while attempting to connect to Hugging Face. Please check your internet connect and logs."))


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
    response = client.voices.get_all()
    tts_voice_list = response.voices
    return [f"{v.name}" for v in tts_voice_list]


def get_edge_tts_voices():
    import edge_tts
    tts_voice_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())
    return [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list if v['ShortName'].startswith('en-')]


def eleven_labs_inference(parent, text, output_file, voice, panel, api=False):
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


def edge_tts_inference(parent, text, output_file, voice, panel, api=False):
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


def rvc_inference(parent, input_file, panel, api=False):
    try:
        parent.tts_engine.run_rvc(input_file)
        if not api:
            QMetaObject.invokeMethod(parent, "updateMediaplayer", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, panel), Q_ARG(str, input_file))
            if cfg.get(cfg.xwm_enabled):
                create_lip_and_fuz(parent, input_file)
            QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception("RVC inference failed")
        if not api:
            QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Generate Audio"), Q_ARG(str, "An Error Occured while attempting to generate audio. Please check your logs and report the issue if needed"))


def xtts_inference(parent, output_file, text, selected_audio, panel, api=False):
    try:
        parent.tts_engine.generate_audio(text=text, voice=selected_audio, language="en", output_file=output_file)
        if not api:
            QMetaObject.invokeMethod(parent, "updateMediaplayer", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, panel), Q_ARG(str, output_file))
            if cfg.get(cfg.xwm_enabled):
                create_lip_and_fuz(parent, output_file)
            QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception("XTTSv2 inference failed")
        if not api:
            QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Generate Audio"), Q_ARG(str, "An Error Occured while attempting to generate audio. Please check your logs and report the issue if needed"))


def gpt_sovits_inference(parent, output_file, text, selected_audio, panel, transcribe_state, api=False):
    try:
        parent.tts_engine.generate_audio(text=text, voice=selected_audio, language="en", output_file=output_file,
                                         transcript=transcribe_state)
        if not api:
            QMetaObject.invokeMethod(parent, "updateMediaplayer", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, panel), Q_ARG(str, output_file))
            if cfg.get(cfg.xwm_enabled):
                create_lip_and_fuz(parent, output_file)
            QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception("GPT_SoVITS inference failed")
        if not api:
            QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Generate Audio"), Q_ARG(str, "An Error Occured while attempting to generate audio. Please check your logs and report the issue if needed"))


def styletts2_inference(parent, output_file, text, selected_audio, panel, api=False):
    try:
        parent.tts_engine.generate_audio(text=text, voice=selected_audio, language="en", output_file=output_file)
        if not api:
            QMetaObject.invokeMethod(parent, "updateMediaplayer", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, panel), Q_ARG(str, output_file))
            if cfg.get(cfg.xwm_enabled):
                create_lip_and_fuz(parent, output_file)
            QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception("StyleTTS2 inference failed")
        if not api:
            QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Generate Audio"), Q_ARG(str, "An Error Occured while attempting to generate audio. Please check your logs and report the issue if needed"))


def voicecraft_inference(parent, output_file, text, selected_audio, panel, start_time, end_time, start_tts_time, transcribe_state, api=False):
    try:
        parent.tts_engine.generate_audio(text=text, voice=selected_audio, language="en", output_file=output_file,
                                         transcribe_state=transcribe_state, prompt_end_time=start_tts_time, edit_start_time=start_time, edit_end_time=end_time)
        if not api:
            QMetaObject.invokeMethod(parent, "updateMediaplayer", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, panel), Q_ARG(str, output_file))
            if cfg.get(cfg.xwm_enabled):
                create_lip_and_fuz(parent, output_file)
            QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception("VoiceCraft inference failed")
        if not api:
            QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Generate Audio"), Q_ARG(str, "An Error Occured while attempting to generate audio. Please check your logs and report the issue if needed"))


def update_progress(parent, total, time_total, count):
    additional_data_points_needed = total - count
    avg_duration = time_total / count
    estimated_duration = (avg_duration * additional_data_points_needed)
    td = timedelta(seconds=estimated_duration)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    QMetaObject.invokeMethod(parent, "update_loader", Qt.QueuedConnection, Q_ARG(str, f"Completed: {count}/{total}. Estimated Duration: {hours:02}:{minutes:02}:{seconds:02}"))

def formatted_time_stamp():
    current_time = datetime.now()
    time_stamp = current_time.strftime("%Y%m%d%H%M%S")
    return f"{time_stamp[:4]}_{time_stamp[4:6]}_{time_stamp[6:8]}_{time_stamp[8:10]}_{time_stamp[10:12]}_{time_stamp[12:14]}"


def formatted_time_stamp_uuid():
    unique_id = uuid.uuid4()
    return f"{formatted_time_stamp()}_{unique_id.hex[:10]}"


def get_bulk_folder(engine_name):
    output_folder = f"bulk_outputs/{formatted_time_stamp()}_{engine_name}"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def process_xwm_file(xwm_file, cfg, files, use_existing_lip=False):
    wav_file = xwm_file.replace(".xwm", ".wav")
    create_xwm(xwm_file, wav_file, encode=False)

    if cfg.get(cfg.replace_existing):
        if os.path.exists(xwm_file):
            os.remove(xwm_file)

    if not use_existing_lip and os.path.exists(xwm_file.replace(".xwm", ".lip")):
        os.remove(xwm_file.replace(".xwm", ".lip"))

    files.add(wav_file)


def process_fuz_file(fuz_file, cfg, files, use_existing_lip=False):
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


def process_rvc_file(tts_engine, wav_file, replace, output_folder, parent, use_existing_lip=False):
    start = datetime.now()
    try:
        existing_lip = None
        if not replace:
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


def bulk_fuz(parent, directory, include_subdir, threads=1):
    wav_files = glob.glob(os.path.join(directory, '**', '*.wav'), recursive=include_subdir)
    xwm_files = glob.glob(os.path.join(directory, '**', '*.xwm'), recursive=include_subdir)
    count = 0
    time_total = 0
    files = set()
    files.update(wav_files)
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
            future = executor.submit(create_lip_and_fuz, parent, wav_file, 44100, True, None)
            futures.append(future)

        for future in as_completed(futures):
            time_taken, fuz_file = future.result()
            time_total += time_taken
            count += 1
            update_progress(parent, total, time_total, count)

    QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))



def bulk_rvc_inference(parent, directory, model, include_subdir, replace, threads=1, use_existing_lip=True):
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

    load_whisper(parent)

    tts_engines = []

    if output_folder is None and not replace:
        output_folder = get_bulk_folder('RVC')
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
            from tts_engines.rvc_engine import RVC_Engine
            tts_engine = RVC_Engine()
            tts_engine.setup(model['name'], has_rvc, not is_trained)
            tts_engine.preload_rvc_params()
            tts_engines.append(tts_engine)

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for idx, wav_file in enumerate(files):
                tts_engine = tts_engines[idx % threads]
                future = executor.submit(process_rvc_file, tts_engine, wav_file, replace, output_folder, parent, use_existing_lip)
                futures.append(future)

            for future in as_completed(futures):
                time_taken = future.result()
                time_total += time_taken
                count += 1
                update_progress(parent, total, time_total, count)

    for engine in tts_engines:
        engine.clean()
        del engine

    if engine_changed:
        QMetaObject.invokeMethod(parent, "afterRVC", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    else:
        QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))


def bulk_inference(parent):
    if parent.tts_engine.engine_name != 'VoiceCraft':
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
                        parent.tts_engine.setup(character, has_rvc, not is_trained)

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


def get_reference(parent, character, reference):
    character_model = parent.characters_data[character] if character in parent.characters_data else None
    if character_model is not None:
        for voice_file in character_model['voicefiles']:
            if reference in voice_file['filename']:
                filename = f"{voice_file['filename']}".replace('.fuz', '')
                if not os.path.exists(f"temp/{filename}.wav"):
                    voice_file['folder'] = character
                    extra_audio_from_bsa(voice_file, filename)
                return f"temp/{filename}.wav"


def get_character_model(character, models, custom_models):
    if character.startswith("custom_") and custom_models is not None:
        return custom_models[character] if character in custom_models else None
    else:
        return models[character] if character in models else None


def get_trained_character(character_model, engine_name):
    rvc = False
    trained = False
    if "RVC" in character_model and character_model['RVC'] is not None:
        rvc = True

    if engine_name in character_model and character_model[engine_name] is not None:
        trained = True

    return trained, rvc


def get_character_models(character, characters_data):
    for character_model in characters_data:
        if character in character_model['name'] and character == character_model['name']:
            return character_model

    return None


def get_latest_release():
    try:
        response = requests.get(config.RELEASE_URL)
        if response.status_code == 200:
            return response.url.split('/')[-1]
        else:
            logger.exception(f"Failed to fetch latest release: {response.status_code}")
            return config.VERSION
    except Exception as e:
        logger.exception(f"Failed to fetch latest release: {e}")
        return config.VERSION


def get_model_diff(old_json, new_json):
    result = []
    if old_json is not None and new_json is not None:
        old_characters = old_json.get('characters', [])
        new_characters = new_json.get('characters', [])

        old_entries = {entry['name']: entry for entry in old_characters}
        new_entries = {entry['name']: entry for entry in new_characters}

        for name in new_entries:
            if name not in old_entries:
                new_entry = new_entries[name]
                new_models = [key for key in new_entry if key not in ['name', 'display_name']]
                result.append(f"Added: {new_entry['display_name']}\n    -Engine: {', '.join(new_models)}")
                continue

            old_entry = old_entries[name]
            new_entry = new_entries[name]

            model_changes = []
            for key in new_entry:
                if key not in old_entry:
                    model_changes.append(f" -Added: {key}")
                elif isinstance(new_entry[key], dict):
                    old_version = old_entry[key].get('version')
                    new_version = new_entry[key].get('version')
                    if old_version != new_version:
                        model_changes.append(f" -Updated: {key} version {new_version}")

            if model_changes:
                changes = "\n ".join(model_changes)
                result.append(f"Changed: {new_entry['display_name']}\n      {changes}")

    if not result:
        return None

    return "\n \n".join(result)


def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
