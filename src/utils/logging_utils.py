from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.FallTalk import FallTalkApp

import logging
import os
import sys
import shutil

from src.utils.filesystem_utils import get_app_root
from logging.handlers import RotatingFileHandler

logger: Optional[logging.Logger] = None

def setup_logging():
    # Configure the logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    os.makedirs(os.path.join(get_app_root(),"logs"), exist_ok=True)

    # Rotate logs before creating new handler
    rotate_logs()

    # Create rotating handler with large file size limit
    handler = logging.handlers.RotatingFileHandler(
        os.path.join(get_app_root(), 'logs/falltalk.log'),
        maxBytes=1024 * 1024 * 50,  # 50MB (safety net for single-run logging)
        backupCount=20,  # Should match our manual rotation
        encoding='utf-8'
    )
    handler.setLevel(logging.DEBUG)

    # Formatter and activation
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Set up stdout and stderr redirection
    stdout_logger = logging.getLogger('stdout')
    stderr_logger = logging.getLogger('stderr')

    # Redirect stdout and stderr to the logger
    sys.stdout = LoggerStream(stdout_logger, logging.INFO)
    sys.stderr = LoggerStream(stderr_logger, logging.ERROR)

    # Return the root logger for convenience
    global logger
    logger = logging.getLogger('falltalk')
    logger.setLevel(logging.DEBUG)

    return root_logger

def rotate_logs():
    log_dir = os.path.join(get_app_root(),"logs")
    main_log = os.path.join(log_dir, "falltalk.log")

    if not os.path.exists(main_log):
        return

    # Remove falltalk.log.20 if it exists
    backup_20 = os.path.join(log_dir, "falltalk.log.20")
    if os.path.exists(backup_20):
        os.remove(backup_20)

    # Shift backups (backwards to avoid conflicts)
    for i in range(19, 0, -1):
        src = os.path.join(log_dir, f"falltalk.log.{i}")
        dst = os.path.join(log_dir, f"falltalk.log.{i+1}")
        if os.path.exists(src):
            shutil.move(src, dst)

    # Move main log to .1
    shutil.move(main_log, os.path.join(log_dir, "falltalk.log.1"))

class LoggerStream:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            try:
                self.logger.log(self.level, message.strip())
            except UnicodeEncodeError:
                # Fallback to a safe encoding
                safe_message = message.encode('utf-8', errors='replace').decode('utf-8')
                self.logger.log(self.level, safe_message.strip())

    def flush(self):
        pass

    def isatty(self):
        return False
