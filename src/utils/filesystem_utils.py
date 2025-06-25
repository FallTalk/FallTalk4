import os
import sys


def get_app_root():
    if getattr(sys, 'frozen', False):
        # Running from PyInstaller bundle
        return os.path.dirname(sys.executable)
    else:
        # Running from normal Python environment
        return os.path.abspath(".")

def get_app_code_root():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    else:
        return get_app_root()


def check_files_in_directory(directory_path):
    try:
        # List all entries in the directory
        entries = os.listdir(directory_path)

        # Filter out directories, keeping only files
        files = [entry for entry in entries
                 if os.path.isfile(os.path.join(directory_path, entry))]

        if not files:
            return False
        return True

    except FileNotFoundError:
        return False
    except PermissionError:
        return False
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False