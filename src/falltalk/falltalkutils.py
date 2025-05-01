# This file is a compatibility layer for the refactored utility functions
# It imports all functions from the new utility modules and re-exports them
# to maintain backward compatibility with the rest of the codebase

import logging

import requests

from src.falltalk import config

# Import all utility functions from the new modules

logger = logging.getLogger('falltalk')
logger.setLevel(logging.DEBUG)

