import logging
import os
import platform
import shutil
import uuid
from datetime import datetime

from src.utils.filesystem_utils import get_app_root

logger = logging.getLogger('falltalk')
logger.setLevel(logging.DEBUG)


def clean_tmp_folder():
    clean_folder(os.path.join(get_app_root(), "temp"))


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


def sanitize_filename(filename):
    invalid_chars = '<>:"/\\|?*%'
    return ''.join(c for c in filename if c not in invalid_chars)


def formatted_time_stamp():
    current_time = datetime.now()
    time_stamp = current_time.strftime("%Y%m%d%H%M%S")
    return f"{time_stamp[:4]}_{time_stamp[4:6]}_{time_stamp[6:8]}_{time_stamp[8:10]}_{time_stamp[10:12]}_{time_stamp[12:14]}"


def formatted_time_stamp_uuid():
    unique_id = uuid.uuid4()
    return f"{formatted_time_stamp()}_{unique_id.hex[:10]}"


def get_bulk_folder(engine_name):
    output_folder = os.path.join(get_app_root(), f"bulk_outputs/{formatted_time_stamp()}_{engine_name}")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def get_output_file_name(file_name, output_dir, model_name, engine_name):
    if file_name is None or file_name == "" or file_name == "Random":
        unique_id = uuid.uuid4()
        file_name = f"{formatted_time_stamp()}_{model_name}_{engine_name}_{unique_id.hex[:10]}"

    file_name = sanitize_filename(file_name)
    path = os.path.abspath(str(os.path.join(output_dir, model_name, f"{file_name}.wav")))
    os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)
    return path


