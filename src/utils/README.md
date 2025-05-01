# FallTalk Utilities

This directory contains utility functions used throughout the FallTalk application, organized into logical groups.

## Overview

The utilities were refactored from the original `falltalkutils.py` file to improve maintainability and organization. The functions are now grouped into the following categories:

- **audio_utils.py**: Functions for audio processing, conversion, and manipulation
- **file_utils.py**: Functions for file operations and path handling
- **huggingface_utils.py**: Functions for downloading models from Hugging Face
- **model_utils.py**: Functions for loading and managing TTS models
- **inference_utils.py**: Functions for TTS inference with various engines
- **bulk_utils.py**: Functions for bulk processing of audio files

## Usage

You can import specific functions directly from their respective modules:

```python
from src.utils.audio_utils import create_lip_and_fuz
from src.utils.model_utils import load_whisper
```

Or you can import all functions from the `src.utils` package:

```python
from src.utils import create_lip_and_fuz, load_whisper
```

For backward compatibility, all functions are also re-exported from the original `src.falltalk.falltalkutils` module:

```python
from src.falltalk.falltalkutils import create_lip_and_fuz, load_whisper
```

## Function Groups

### Audio Utils

Functions for audio processing, conversion, and manipulation:
- `extra_audio_from_bsa`: Extract audio from BSA archives
- `create_fuz_files`: Create FUZ files for Fallout
- `create_lip_and_fuz`: Create LIP and FUZ files for Fallout
- `create_lip_files`: Create LIP files for Fallout
- `create_xwm`: Create XWM files for Fallout
- `extract_fuz`: Extract content from FUZ files
- `extract_bsa`: Extract content from BSA archives
- `combine_wav_files`: Combine multiple WAV files into one
- `is_stereo`: Check if audio is stereo
- `load_audio`: Load audio files with resampling

### File Utils

Functions for file operations and path handling:
- `clean_folder`: Clean a folder by removing all files and subdirectories
- `clean_path`: Clean a path string
- `sanitize_filename`: Sanitize a filename
- `formatted_time_stamp`: Generate a formatted timestamp
- `formatted_time_stamp_uuid`: Generate a formatted timestamp with UUID
- `get_bulk_folder`: Get a folder for bulk outputs

### Hugging Face Utils

Functions for downloading models from Hugging Face:
- `download_model_from_hub`: Download a model from Hugging Face
- `downloadXTTS`, `downloadRVC`, etc.: Download specific models
- `download_rvc_models`: Download RVC models
- `download_models`: Download models for a character

### Model Utils

Functions for loading and managing TTS models:
- `load_model`: Load a model
- `get_character_model`: Get a character model
- `get_trained_character`: Check if a character is trained
- `get_character_models`: Get character models
- `load_xtts`, `load_whisper`, etc.: Load specific TTS engines
- `seed_everything`: Set random seeds for reproducibility

### Inference Utils

Functions for TTS inference with various engines:
- `do_transcribe`: Transcribe audio
- `replace_numbers_with_words`: Replace numbers with words in text
- `get_eleven_labs_voices`: Get Eleven Labs voices
- `get_edge_tts_voices`: Get Edge TTS voices
- `eleven_labs_inference`, `edge_tts_inference`, etc.: Inference with specific engines

### Bulk Utils

Functions for bulk processing of audio files:
- `update_progress`: Update progress for bulk operations
- `get_reference`: Get a reference audio file
- `process_xwm_file`, `process_fuz_file`, etc.: Process specific file types
- `bulk_fuz`, `bulk_rvc_inference`, `bulk_inference`: Bulk processing functions