import os.path
import shutil
import threading
import json
import copy
from typing import Optional

import uvicorn
from PySide6.QtCore import Qt, QMetaObject, Q_ARG
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, FileResponse

from src.config.config import cfg, VERSION
from src.enums.engine_type import EngineType
from src.tts_engines import tts_engine
from src.utils.file_utils import sanitize_filename, formatted_time_stamp
from src.utils.logging_utils import logger
from src.utils.inference_utils import (
    do_transcribe, eleven_labs_inference, edge_tts_inference, generic_inference,
    rvc_inference, do_transcribe_before_gen, preprocess_text
)
from src.utils.model_utils import (
    load_model, load_xtts, load_gpt_sovits, load_dia, load_rvc, load_spark,
    load_fish, load_f5, load_llasa, load_orpheus, load_style_tts2, load_upscaler, load_csm,
    load_higgs, load_chatterbox, load_dmo_speech2
)
from tts_engines.whisper_engine import Whisper_Engine

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

        # Dictionary to store engine load functions
        self.engine_load_functions = {
            EngineType.XTTS_V2: load_xtts,
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
            EngineType.DMOSPEECH2: load_dmo_speech2
        }

        # Load static JSON data
        self._load_static_json_data()

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

        if engine_type in self.engine_load_functions:
            self.engine_load_functions[engine_type](self)
            return True
        else:
            self.tts_engine = None
            return False

    def after_engine_load(self, parent, engine):
        """Callback after engine is loaded"""
        logger.debug(f"Engine {engine} loaded")

    def afterModelLoader(self, parent):
        """Callback after model is loaded"""
        logger.debug(f"Model {self.tts_engine.model_name} loaded")

    def afterGen(self, parent):
        """Callback after audio generation"""
        logger.debug("Audio generation completed")

    def onError(self, parent, title, text):
        """Callback for errors"""
        logger.error(f"{title}: {text}")

    def onWarn(self, parent, title, text):
        """Callback for warnings"""
        logger.warning(f"{title}: {text}")

    def load_model(self, character, rvc=None, display_name=None, base_model=False, model_engine_version=None):
        """Load a model for a character"""
        if character:
            load_model(self, character, rvc, display_name, base_model, model_engine_version)
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

    def __init__(self, falltak_app=None):
        self.falltak_app = falltak_app
        self.server_thread = FastAPIServer()
        self.server_thread.start()
        if falltak_app:
            self.falltak_app.openapi = self.custom_openapi()

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

            # If we have a FallTalk app instance, use it
            if self.falltak_app:
                QMetaObject.invokeMethod(self.falltak_app, "engine_change", Qt.QueuedConnection, Q_ARG(str, data['engine']))
                return JSONResponse({"new_engine": data['engine'], "previous_engine": cfg.get(cfg.engine)})

            # Otherwise use the server thread's engine loading
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

        # If we have a FallTalk app instance, use it
        if self.falltak_app:
            self.falltak_app.load_trained_model(character, None, rvc)
            return JSONResponse({"model_loaded": character})

        # Otherwise use the server thread's model loading
        else:
            success = self.server_thread.load_model(character, rvc, display_name, base_model, model_engine_version)
            return JSONResponse({
                "model_loaded": character,
                "success": success
            })

    @app.post('/inference')
    async def api_inference(self, data: dict):
        # Validate required fields
        if 'text' not in data and 'input_file' not in data:
            return JSONResponse({"error": "Either text or input_file must be provided"}, status_code=400)

        # Get output file
        if 'output_file' not in data:
            if self.falltak_app:
                data['output_file'] = self.falltak_app.get_output_file_name(None)
            else:
                data['output_file'] = self.server_thread.get_output_file_name(None)

        # Get current engine
        engine = cfg.get(cfg.engine)
        engine_type = EngineType(engine)

        # Process based on engine type
        if engine_type == EngineType.RVC:
            if 'input_file' not in data:
                return JSONResponse({"error": "input_file is required for RVC"}, status_code=400)

            shutil.copy(data['input_file'], data['output_file'])

            if self.falltak_app:
                rvc_inference(self.falltak_app, data['input_file'], None, True)
            else:
                rvc_inference(self.server_thread, data['input_file'], None, True)

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

            if self.falltak_app:
                generic_inference(
                    self.falltak_app,
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
            else:
                generic_inference(
                    self.server_thread,
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

        if self.falltak_app:
            result = do_transcribe(self.falltak_app, audio_file, None, True)
        else:
            result = do_transcribe(self.server_thread, audio_file, None, True)

        return JSONResponse({"transcript": result})

    @app.post('/cfg/update')
    async def api_cfg_update(self, data: dict):
        for key, value in data.items():
            if hasattr(cfg, key):
                cfg.set(getattr(cfg, key), value)
            else:
                logger.warning(f"Unknown config key: {key}")

        return JSONResponse({"config_save": True})
