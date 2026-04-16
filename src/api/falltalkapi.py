from __future__ import annotations

import json
import os.path
import shutil
import threading
from typing import Optional, TYPE_CHECKING

import uvicorn
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, FileResponse

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState

from src.config.config import cfg, VERSION
from src.enums.engine_type import EngineType
from src.tts_engines import tts_engine
from src.utils.file_utils import sanitize_filename, formatted_time_stamp
from src.utils.inference_utils import (
    do_transcribe, generic_inference,
    rvc_inference, preprocess_text
)
from src.utils.logging_utils import logger
from src.utils.model_utils import (
    load_model, load_gpt_sovits, load_dia, load_rvc, load_spark,
    load_fish, load_f5, load_llasa, load_orpheus, load_style_tts2, load_csm,
    load_higgs, load_chatterbox, load_dmo_speech2, load_vibe, load_qwen
)
from src.tts_engines.whisper_engine import Whisper_Engine

app = FastAPI()


class FastAPIServer(threading.Thread):

    def __init__(self):
        super().__init__()

        self.config = None
        self.server = None
        self.tts_engine: Optional[tts_engine] = None
        self.transcription_engine: Optional[Whisper_Engine] = None
        self.characters_data = None
        self.models = None
        self.shared_models = None
        self.custom_models = None
        self.default_references = None
        self._callbacks = None
        self._state = None

        # Dictionary to store engine load functions
        self.engine_load_functions = {
            EngineType.GPT_SOVITS: load_gpt_sovits,
            EngineType.STYLE_TTS2: load_style_tts2,
            EngineType.DIA: load_dia,
            EngineType.LLASA: load_llasa,
            EngineType.ORPHEUS: load_orpheus,
            EngineType.FISH_SPEECH: load_fish,
            EngineType.F5: load_f5,
            EngineType.RVC: load_rvc,
            EngineType.SPARK: load_spark,
            EngineType.CSM: load_csm,
            EngineType.HIGGS: load_higgs,
            EngineType.CHATTERBOX: load_chatterbox,
            EngineType.DMOSPEECH2: load_dmo_speech2,
            EngineType.VIBE: load_vibe,
            EngineType.QWEN3_TTS: load_qwen,
        }

        # Load static JSON data
        self._load_static_json_data()

    def _ensure_callbacks(self):
        """Ensure standalone callbacks exist for headless API mode."""
        if self._callbacks is None:
            from src.ui_imgui.state import AppCallbacks, AppState

            state = AppState()
            state.tts_engine = self.tts_engine
            state.transcription_engine = self.transcription_engine
            state.characters_data = self.characters_data or {}
            state.models = self.models or {}
            state.custom_models = self.custom_models
            state.default_references = self.default_references

            self._callbacks = AppCallbacks(
                on_progress=lambda msg: logger.debug(f"Progress: {msg}"),
                on_done=lambda: logger.debug("Operation complete"),
                on_error=lambda title, msg: logger.error(f"{title}: {msg}"),
                on_warn=lambda title, msg: logger.warning(f"{title}: {msg}"),
                on_model_loaded=lambda: logger.debug(f"Model loaded"),
                on_media_update=lambda path: logger.debug(f"Media updated: {path}"),
                on_continue_load=lambda: logger.debug("Continue load"),
                state_ref=state,
            )
            self._state = state

        # Keep state in sync with server attributes
        self._state.tts_engine = self.tts_engine
        self._state.transcription_engine = self.transcription_engine
        return self._callbacks

    def _load_static_json_data(self):
        """Load static JSON data from files and cache it in memory"""
        if self.characters_data is None:
            with open(os.path.join(os.path.abspath("config/characters.json")), 'r', encoding='utf-8') as file:
                self.characters_data = {character['name']: character for character in json.load(file)}

        if self.models is None:
            # Load character-specific models from modelsv2.json
            with open(os.path.join(os.path.abspath("config/modelsv2.json")), 'r', encoding='utf-8') as file:
                self.models = {model['name']: model for model in json.load(file)['characters']}

        if self.shared_models is None:
            # Load shared models from sharedmodels.json if it exists
            shared_models_path = os.path.join(os.path.abspath("config/sharedmodels.json"))
            if os.path.exists(shared_models_path):
                with open(shared_models_path, 'r', encoding='utf-8') as file:
                    self.shared_models = json.load(file)

                    # Process each shared model
                    for shared_model in self.shared_models:
                        engine = shared_model.get('engine')
                        engine_version = shared_model.get('engine_version', '1')
                        model_version = shared_model.get('model_version', '1')
                        model_name = shared_model.get('model_name')
                        model_type = shared_model.get('model_type', 'safetensors')
                        characters = shared_model.get('characters', [])

                        # Add the shared model to each character's models
                        for character_name in characters:
                            if character_name in self.models:
                                self.models[character_name][engine] = {
                                    'version': model_version,
                                    'engine_version': engine_version,
                                    'engine': engine,
                                    'type': model_type,
                                    'is_shared': True,
                                    'shared_model_name': model_name,
                                    'characters': characters
                                }

        # Load default references from default_references.json if it exists
        default_references_path = os.path.join(os.path.abspath("config/default_references.json"))
        if os.path.exists(default_references_path):
            with open(default_references_path, 'r', encoding='utf-8') as file:
                self.default_references = json.load(file)
        else:
            self.default_references = None

    def _load_custom_models(self):
        """Load custom models from file - this can change during runtime"""
        if os.path.exists(os.path.join('config', 'custom_models.json')):
            with open(os.path.join(os.path.abspath("config/custom_models.json")), 'r', encoding="utf-8") as file:
                self.custom_models = {model['name']: model for model in json.load(file)}
        else:
            self.custom_models = None

    def run(self):
        self.config = uvicorn.Config(app, host="127.0.0.1", port=2277)
        self.server = uvicorn.Server(self.config)
        self.server.run()

    def shutdown(self):
        if self.tts_engine is not None:
            self.tts_engine.clean()
        self.server.should_exit = True
        self.server.force_exit = True
        self.server.shutdown()

    def load_engine(self, engine_type):
        """Load a TTS engine"""
        logger.debug(f"Loading engine {engine_type.value}")

        if self.tts_engine is not None:
            self.tts_engine.clean()

        callbacks = self._ensure_callbacks()

        if engine_type in self.engine_load_functions:
            self.engine_load_functions[engine_type](callbacks)
            # Sync engine back from state
            self.tts_engine = self._state.tts_engine
            return True
        else:
            self.tts_engine = None
            return False

    def load_model_standalone(self, character, rvc=None, display_name=None, base_model=False, model_engine_version=None):
        """Load a model for a character (standalone mode)"""
        callbacks = self._ensure_callbacks()
        if character:
            load_model(callbacks, character, rvc, display_name, base_model, model_engine_version)
            return True
        return False

    def get_output_file_name(self, file_name):
        """Generate an output file name"""
        if file_name is None or file_name == "" or file_name == "Random":
            file_name = f"{formatted_time_stamp()}_{self.tts_engine.model_name}_{self.tts_engine.engine_name}"

        file_name = sanitize_filename(file_name)
        path = os.path.abspath(os.path.join(cfg.get(cfg.output_dir), self.tts_engine.model_name, f"{file_name}.wav"))
        os.makedirs(os.path.join(cfg.get(cfg.output_dir), self.tts_engine.model_name), exist_ok=True)
        return path

    def combine_references(self, references):
        """Combine multiple reference audio files into one"""
        from src.utils.audio_utils import combine_wav_files

        if not references:
            return None
        elif len(references) == 1:
            logger.debug(f"Using single reference: {references[0]}")
            return references[0]
        else:
            logger.debug(f"Combining references: {references}")
            return combine_wav_files(references)


class FallTalkAPI:

    def __init__(self, app_state: Optional[AppState] = None):
        self.app_state = app_state
        self.server_thread = FastAPIServer()
        self.server_thread.start()

    def custom_openapi(self):
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title="FallTalk",
            version=VERSION,
            description="FallTalk API",
            routes=app.routes,
        )
        openapi_schema["info"]["x-logo"] = {
            "url": "http://127.0.0.1:2277/logo"
        }
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    def shutdown(self):
        self.server_thread.shutdown()

    @app.get("/logo", include_in_schema=False)
    async def get_logo(self):
        return FileResponse(os.path.abspath("resource/falltalk.png"))

    @app.post('/engine/load')
    async def api_engine_load(self, data: dict):
        if 'engine' in data:
            engine_type = EngineType(data['engine'])

            # If we have an AppState (GUI mode), use the command queue
            if self.app_state:
                self.app_state.api_command_queue.put({"action": "engine_change", "engine": data['engine']})
                return JSONResponse({"new_engine": data['engine'], "previous_engine": cfg.get(cfg.engine)})

            # Otherwise use the server thread's engine loading (standalone mode)
            else:
                cfg.set(cfg.engine, data['engine'])
                success = self.server_thread.load_engine(engine_type)
                return JSONResponse({
                    "new_engine": data['engine'],
                    "previous_engine": cfg.get(cfg.engine),
                    "success": success
                })

        return JSONResponse({"error": "No engine specified"}, status_code=400)

    @app.post('/model/load')
    async def api_model_load(self, data: dict):
        if 'character' not in data:
            return JSONResponse({"error": "No character specified"}, status_code=400)

        character = data['character']
        rvc = data.get('rvc', None)
        display_name = data.get('display_name', None)
        base_model = data.get('base_model', False)
        model_engine_version = data.get('model_engine_version', None)

        # If we have an AppState (GUI mode), use the command queue
        if self.app_state:
            self.app_state.api_command_queue.put({
                "action": "load_model",
                "character": character,
                "rvc": rvc,
                "display_name": display_name,
                "base_model": base_model,
                "model_engine_version": model_engine_version,
            })
            return JSONResponse({"model_loaded": character})

        # Otherwise use the server thread's model loading (standalone mode)
        else:
            success = self.server_thread.load_model_standalone(character, rvc, display_name, base_model, model_engine_version)
            return JSONResponse({
                "model_loaded": character,
                "success": success
            })

    @app.post('/inference')
    async def api_inference(self, data: dict):
        # Validate required fields
        if 'text' not in data and 'input_file' not in data:
            return JSONResponse({"error": "Either text or input_file must be provided"}, status_code=400)

        # Get the callbacks to use
        if self.app_state and self.app_state.callbacks:
            callbacks = self.app_state.callbacks
        else:
            callbacks = self.server_thread._ensure_callbacks()

        # Get output file
        if 'output_file' not in data:
            data['output_file'] = self.server_thread.get_output_file_name(None)

        # Get current engine
        engine = cfg.get(cfg.engine)
        engine_type = EngineType(engine)

        # Process based on engine type
        if engine_type == EngineType.RVC:
            if 'input_file' not in data:
                return JSONResponse({"error": "input_file is required for RVC"}, status_code=400)

            shutil.copy(data['input_file'], data['output_file'])
            rvc_inference(callbacks, data['input_file'], None, True)

        else:
            # For all other engines, use generic_inference
            if 'text' not in data:
                return JSONResponse({"error": "text is required for this engine"}, status_code=400)

            text = preprocess_text(data['text'])

            # Handle reference audio (single file or list of files)
            reference_audio = data.get('reference_audio', None)
            if isinstance(reference_audio, list) and len(reference_audio) > 0:
                reference_audio = self.server_thread.combine_references(reference_audio)

            transcript = data.get('transcript', None)
            start_time = data.get('start_time', None)
            end_time = data.get('end_time', None)
            speaker = data.get('speaker', None)

            # Create transcribe state if transcript is provided
            transcribe_state = {'transcript': transcript} if transcript else None

            # Check if engine requires reference audio
            engine_type = EngineType(engine)
            if (engine_type.needs_reference_when_trained or
                (self.server_thread.tts_engine and self.server_thread.tts_engine.is_base)) and not reference_audio:
                return JSONResponse({"error": "Reference audio is required for this engine"}, status_code=400)

            generic_inference(
                callbacks,
                data['output_file'],
                text,
                reference_audio,
                None,
                transcribe_state,
                start_time,
                end_time,
                True,
                speaker
            )

        # Return response
        if data.get('stream', False):
            return FileResponse(data['output_file'], media_type="audio/wav")
        else:
            return JSONResponse({"output": data['output_file']})

    @app.post('/transcribe')
    async def api_transcribe(self, data: dict):
        if 'audio_file' not in data:
            return JSONResponse({"error": "audio_file is required"}, status_code=400)

        audio_file = data['audio_file']

        if not os.path.exists(audio_file):
            return JSONResponse({"error": f"File {audio_file} not found"}, status_code=404)

        if self.app_state and self.app_state.callbacks:
            callbacks = self.app_state.callbacks
        else:
            callbacks = self.server_thread._ensure_callbacks()

        result = do_transcribe(callbacks, audio_file, None, True)

        return JSONResponse({"transcript": result})

    @app.post('/cfg/update')
    async def api_cfg_update(self, data: dict):
        for key, value in data.items():
            if hasattr(cfg, key):
                cfg.set(getattr(cfg, key), value)
            else:
                logger.warning(f"Unknown config key: {key}")

        return JSONResponse({"config_save": True})
