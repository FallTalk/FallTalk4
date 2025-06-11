import glob
import os
from abc import ABC, abstractmethod
from typing import Optional

import librosa
import numpy as np
import torch
import soundfile as sf

from enums.engine_type import EngineType
from src.utils import logging_utils
from src.config.config import cfg
from src.utils.filesystem_utils import get_app_root


class tts_engine(ABC):
    def __init__(self):
        self.model = None
        self.model_name = None
        self.shared_model_name = None
        self.engin_type: Optional[EngineType] = None
        self.model_path = None
        self.model_engine_version = None
        self.model_type = 'safetensors'
        self.rvc_model = False
        self.is_base = False
        self.is_shared = False
        self.device = cfg.get(cfg.device)
        self.rvc_pipeline = None
        self.rvc_preload = False
        self.rvc_parameters = None
        self.rvc_pth_path = None
        self.rvc_index_path = None
        self.rvc_model_version = None
        self.apbwe_engine = None
        self.characters = []

    def get_model(self, engin_type: EngineType, model_type=None, model_engine_version=None, shared_model_name=None):
        if(model_type is None):
            model_type = self.model_type
        if(model_engine_version is None):
            model_engine_version = self.model_engine_version

        # If this is a shared model, use the shared model name for the path
        if self.is_shared and shared_model_name:
            directory = os.path.join(get_app_root(), "models", "shared", engin_type.get_model_path(self.shared_model_name, model_engine_version))
        else:
            directory = os.path.join(get_app_root(), "models", engin_type.get_model_path(self.model_name, model_engine_version))

        if engin_type.loads_from_dir:
            return directory
        else:
            model_files = glob.glob(os.path.join(directory, '*.' + model_type))
            if not model_files:
                print(f"No model files found in the directory: {directory}")
            else:
                return max(model_files, key=os.path.getmtime)
            pass

    def basic_unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None

        self.model_name = None
        self.model_path = None
        self.model_engine_version = None
        self.rvc_model = False
        self.is_base = False
        self.is_shared = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def setup(self, selected_model, rvc=False, base_model=False, model_version=None, is_shared=False, shared_model_name=None, characters=None):
        if characters is None:
            characters = []

        from src.tts_engines.rvc.infer.infer import RVCPipeline

        print(f"setup {selected_model} version {model_version} shared: {is_shared} shared_model_name: {shared_model_name}")
        self.rvc_model = rvc
        self.is_shared = is_shared

        if selected_model is not None and not base_model and not is_shared:
            if self.model_name != selected_model:
                if self.model:
                    self.unload_model()
                self.model_name = selected_model
                self.model_engine_version = model_version

                self.model_path = self.get_model(self.engin_type, self.model_type)
                self.load_model()
        elif is_shared:
            if self.model and self.shared_model_name == shared_model_name and self.model_engine_version == model_version and selected_model in self.characters:
                print("Same Shared Model, no need to unload!")
                self.model_name = selected_model
            else:
                self.unload_model()
                self.characters = characters
                self.is_shared = True
                self.model_name = selected_model
                self.shared_model_name = shared_model_name
                self.model_engine_version = model_version
                self.model_path = self.get_model(self.engin_type, self.model_type, shared_model_name=shared_model_name)
                self.load_model()


        elif base_model:
            if self.model_name is not None and not self.is_base:
                self.unload_model()

            if self.model_name is None or self.model_name != selected_model:
                self.is_base = True
                self.model_name = selected_model
                self.load_model()
            else:
                print("reusing model")

        if self.rvc_model:
            if self.rvc_pipeline is not None:
                self.rvc_pipeline.clean_up()

            self.rvc_pipeline = RVCPipeline(cfg.get(cfg.device))

            # Handle RVC models for shared models
            if self.is_shared and shared_model_name:
                self.rvc_pth_path = os.path.join(get_app_root(), "models", "shared", "RVC", shared_model_name, "model.pth")
                self.rvc_index_path = os.path.join(get_app_root(), "models", "shared", "RVC", shared_model_name, "model.index")
            else:
                self.rvc_pth_path = self.get_model(EngineType.RVC, "pth", "1")
                self.rvc_index_path = self.get_model(EngineType.RVC, "index", "1")

            if os.path.isfile(self.rvc_pth_path) and os.path.isfile(self.rvc_index_path):
                self.rvc_pipeline.load_person(self.rvc_pth_path)
                self.rvc_pipeline.load_index_file(self.rvc_index_path, cfg.get(cfg.rvc_training_data_size))


    def handle_lowvram_change(self):
        if torch.cuda.is_available():
            if self.device == "cuda":
                self.device = "cpu"
                self.model.to(self.device)
                torch.cuda.empty_cache()
            elif self.device == "cpu":
                self.device = "cuda"
                self.model.to(self.device)

    def handle_deepspeed_change(self, value):
        if value:
            # DeepSpeed enabled
            self.unload_model()
            self.setup(self.model_name)
        else:
            # DeepSpeed disabled
            self.unload_model()
            self.setup(self.model_name)
        return value

    @abstractmethod
    def load_model(self):
        """LOAD"""
        pass

    def clean(self):
        if self.rvc_pipeline is not None:
            self.rvc_pipeline.clean_up()

    @abstractmethod
    def unload_model(self):
        """UNLOAD"""
        pass

    def preload_rvc_params(self):
        self.rvc_preload = True
        self.rvc_parameters = self.get_rvc_params()

    def get_rvc_params(self):
        from src.tts_engines.rvc.infer.infer import RVCParameters
        params = RVCParameters()
        params.f0up_key = cfg.get(cfg.rvc_pitch)
        params.filter_radius = cfg.get(cfg.rvc_filter_radius) / 100.0
        params.index_rate = cfg.get(cfg.rvc_index_influence) / 100.0
        params.rms_mix_rate = cfg.get(cfg.rvc_volume_envelope) / 100.0
        params.protect = cfg.get(cfg.rvc_protect) / 100.0
        params.hop_length = cfg.get(cfg.rvc_hop_length)
        params.f0method = cfg.get(cfg.rvc_pitch_extraction).value
        params.split_audio = cfg.get(cfg.rvc_split_audio)
        params.f0autotune = cfg.get(cfg.rvc_autotune)
        params.embedder_model = cfg.get(cfg.rvc_embedder_model)
        params.training_data_size = cfg.get(cfg.rvc_training_data_size)
        params.pth_path = self.rvc_pth_path
        params.index_path = self.rvc_index_path
        return params

    def process_audio(self, audio_data: np.ndarray, sample_rate: int, output_file: str):
        apbwe_enabled = cfg.get(cfg.apbwe_enabled)
        if apbwe_enabled and self.apbwe_engine:
            audio_data, sample_rate = self.apbwe_engine.upscale(audio_data, sample_rate)

        rvc_enabled = cfg.get(cfg.rvc_enabled)
        if rvc_enabled and self.rvc_model:
            # Process audio through RVC
            audio_data, sample_rate = self.run_rvc(audio_data, sample_rate)

        if not np.issubdtype(audio_data.dtype, np.floating):
            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

        # Ensure sample rate is 44100 Hz if needed
        if sample_rate != 44100:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=44100)
            sample_rate = 44100

        # Write processed audio to output file
        if output_file is not None:
            sf.write(output_file, audio_data, sample_rate, subtype='PCM_16')

    def run_rvc_file(self, input_tts_path):
        audio_data, sample_rate = librosa.load(input_tts_path, sr=48000)
        return self.run_rvc(audio_data, sample_rate)

    def run_rvc(self, audio_data: np.ndarray, sample_rate: int):
        """
        Process audio data through RVC

        Args:
            audio_data (numpy.ndarray): Audio data as numpy array
            sample_rate (int): Sample rate of the audio data

        Returns:
            tuple: (processed_audio, output_sample_rate)
        """
        logging_utils.logger.debug(f"Running RVC on audio data with sample rate {sample_rate}")
        if self.rvc_preload:
            params = self.rvc_parameters
        else:
            params = self.get_rvc_params()

        if not os.path.isfile(params.pth_path) or not os.path.isfile(params.index_path):
            print(f"Model file {params.pth_path} or {params.index_path} does not exist. Exiting.")
            return audio_data, sample_rate

        return self.rvc_pipeline.infer_pipeline(params.f0up_key, params.filter_radius, params.index_rate, params.rms_mix_rate, params.protect, params.hop_length, params.f0method,
                                         audio_data, sample_rate, params.pth_path, params.index_path, params.split_audio, params.f0autotune, params.embedder_model,
                                         params.training_data_size, False)
