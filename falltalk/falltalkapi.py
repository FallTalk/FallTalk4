import os.path
import shutil
import threading

import uvicorn
from PySide6.QtCore import Qt, QMetaObject, Q_ARG
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, FileResponse

import config
from falltalk import falltalkutils
from falltalk.config import cfg

app = FastAPI()


class FastAPIServer(threading.Thread):

    def __init__(self):
        super().__init__()

        self.config = None
        self.server = None

    def run(self):
        self.config = uvicorn.Config(app, host="127.0.0.1", port=2277)
        self.server = uvicorn.Server(self.config)
        self.server.run()

    def shutdown(self):
        self.server.should_exit = True
        self.server.force_exit = True
        self.server.shutdown()


class FallTalkAPI:

    def __init__(self, falltak_app):
        self.falltak_app = falltak_app
        self.server_thread = FastAPIServer()
        self.server_thread.start()
        self.falltak_app.openapi = self.custom_openapi()

    def custom_openapi(self):
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title="FallTalk",
            version=config.VERSION,
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
            QMetaObject.invokeMethod(self.falltak_app, "engine_change", Qt.QueuedConnection, Q_ARG(str, data['engine']))

        return JSONResponse({"new_engine": data['engine'], "previous_engine": cfg.get(cfg.engine)})

    @app.post('/model/load')
    async def api_model_load(self, data: dict):
        if 'model' in data:
            self.falltak_app.onEngineChange(data['engine'])
        return JSONResponse({"model loaded": data['engine']})

    @app.post('/inference')
    async def api_inference(self, data: dict):
        if 'output_file' not in data:
            data['output_file'] = self.falltak_app.get_output_file_name(None)

        engine = cfg.get(cfg.engine)
        if engine == 'RVC':
            shutil.copy(data['input_file'], data['output_file'])
            falltalkutils.rvc_inference(self.falltak_app, data['input_file'], None, True)
        elif engine == 'VoiceCraft':
            falltalkutils.voicecraft_inference(self.falltak_app, data['output_file'], data['text'], data['reference_audio'], None, data['start_time'], data['end_time'], data['tts_start_time'], data['transcribe_state'], True)
        elif engine == 'GPT_SoVITS':
            falltalkutils.gpt_sovits_inference(self.falltak_app, data['output_file'], data['text'], data['reference_audio'], None, data['transcript'], True)
        elif engine == 'StyleTTS2':
            falltalkutils.styletts2_inference(self.falltak_app, data['output_file'], data['text'], data['reference_audio'], None, True)
        elif engine == 'XTTS':
            falltalkutils.xtts_inference(self.falltak_app, data['output_file'], data['text'], data['reference_audio'], None, True)

        if data['stream']:
            return FileResponse(data['output_file'], media_type="audio/wav")
        else:
            return JSONResponse({"output": data['engine']})

    @app.post('/cfg/update')
    async def api_cfg_update(self, data: dict):
        for key, value in data.items():
            cfg.set(cfg['key'], value)

        return JSONResponse({"config_save": True})
