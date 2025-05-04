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