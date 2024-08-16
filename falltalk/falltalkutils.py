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

import PySide6
import ffmpeg
import huggingface_hub
import numpy as np
import requests
import soundfile as sf
from PySide6.QtCore import QMetaObject, QUrl, Qt, Q_ARG
from num2words import num2words

import config
from falltalk.config import cfg, REPO

logger = logging.getLogger('falltalk')



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
    except subprocess.CalledProcessError as e:
        logger.exception("Conversion failed: %s", e.stderr)


def export_xwm(file_name):
    base = os.path.splitext(file_name)[0]
    new_filename = base + ".xwm"
    create_xwm(file_name, new_filename)


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
    except Exception as e:
        logger.exception("Transcription failed")
        if not api:
            QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Transcribe Audio"), Q_ARG(str, "An Error Occured while attempting to transcribe audio. Please check your logs and report the issue if needed"))


def load_model(parent, character=None, rvc=None, display_name=None, base_model=False):
    print(f"load_model {parent} {character}")
    try:
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


def download_models(parent, character, model, rvc):
    try:
        download_model_from_hub(character, model)
        download_rvc_models(character, rvc)
        if model['engine'] == "GPT_SoVITS":
            model['type'] = "ckpt"
            download_model_from_hub(character, model)



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
        QMetaObject.invokeMethod(parent, "afterXtts", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
        QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Engine"), Q_ARG(str, "An Error Occured while loading the engine. Please check your logs and report the issue if needed"))


def load_voicecraft(parent):
    try:
        downloadVoiceCraft(parent)
        downloadRVC(parent)
        from tts_engines.voicecraft_engine import VoiceCraft_Engine
        from tts_engines.whisper_engine import Whisper_Engine
        parent.tts_engine = VoiceCraft_Engine()
        if parent.transcription_engine is None:
            from tts_engines.whisper_engine import Whisper_Engine
            parent.transcription_engine = Whisper_Engine()
        print("VoiceCraft Loaded")
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
        if parent.transcription_engine is None:
            from tts_engines.whisper_engine import Whisper_Engine
            parent.transcription_engine = Whisper_Engine()
        print("GPT_SoVITS Loaded")
        QMetaObject.invokeMethod(parent, "afterGPT_SoVITS", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
        QMetaObject.invokeMethod(parent, "continueLoad", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception(f"Error: {e}")
        QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Load Engine"), Q_ARG(str, "An Error Occured while loading the engine. Please check your logs and report the issue if needed"))


def load_rvc(parent):
    try:
        downloadRVC(parent)
        from tts_engines.rvc_engine import RVC_Engine
        parent.tts_engine = RVC_Engine()
        print("RVC Loaded")
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
            QMetaObject.invokeMethod(parent, "updateMediaplayer", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, panel),  Q_ARG(str, input_file))
            if cfg.get(cfg.xwm_enabled):
                export_xwm(input_file)
            QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception("RVC inference failed")
        if not api:
            QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Generate Audio"), Q_ARG(str, "An Error Occured while attempting to generate audio. Please check your logs and report the issue if needed"))


def xtts_inference(parent, output_file, text, selected_audio, panel, api=False):
    try:
        parent.tts_engine.generate_audio(text=text, voice=selected_audio, language="en", output_file=output_file)
        if not api:
            QMetaObject.invokeMethod(parent, "updateMediaplayer", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, panel),  Q_ARG(str, output_file))
            if cfg.get(cfg.xwm_enabled):
                export_xwm(output_file)
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
            QMetaObject.invokeMethod(parent, "updateMediaplayer", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, panel),  Q_ARG(str, output_file))
            if cfg.get(cfg.xwm_enabled):
                export_xwm(output_file)
            QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception("GPT_SoVITS inference failed")
        if not api:
            QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Generate Audio"), Q_ARG(str, "An Error Occured while attempting to generate audio. Please check your logs and report the issue if needed"))


def styletts2_inference(parent, output_file, text, selected_audio, panel, api=False):
    try:
        parent.tts_engine.generate_audio(text=text, voice=selected_audio, language="en", output_file=output_file)
        if not api:
            QMetaObject.invokeMethod(parent, "updateMediaplayer", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, panel),  Q_ARG(str, output_file))
            if cfg.get(cfg.xwm_enabled):
                export_xwm(output_file)
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
            QMetaObject.invokeMethod(parent, "updateMediaplayer", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, panel),  Q_ARG(str, output_file))
            if cfg.get(cfg.xwm_enabled):
                export_xwm(output_file)
            QMetaObject.invokeMethod(parent, "afterGen", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent))
    except Exception as e:
        logger.exception("VoiceCraft inference failed")
        if not api:
            QMetaObject.invokeMethod(parent, "onError", Qt.QueuedConnection, Q_ARG(PySide6.QtCore.QObject, parent), Q_ARG(str, "Unable to Generate Audio"), Q_ARG(str, "An Error Occured while attempting to generate audio. Please check your logs and report the issue if needed"))


def get_latest_release():
    try:
        response = requests.get(config.RELEASE_URL)
        if response.status_code == 200:
            return response.json()['tag_name']
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
