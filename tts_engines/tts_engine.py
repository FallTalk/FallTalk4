import glob
import os
from abc import ABC, abstractmethod
import torch

from tts_engines.rvc.infer.infer import RVCPipeline, RVCParameters
from falltalk.config import cfg


class tts_engine(ABC):
    def __init__(self):
        self.model = None
        self.model_name = None
        self.engine_name = None
        self.model_path = None
        self.model_type = 'pth'
        self.rvc_model = False
        self.is_base = False
        self.device = cfg.get(cfg.device)
        self.rvc_pipeline = None
        self.rvc_preload = False
        self.rvc_parameters = None
        self.rvc_pth_path = None
        self.rvc_index_path = None

    def get_model(self, engine, model_type):
        directory = os.path.join("models", self.model_name, engine)
        model_files = glob.glob(os.path.join(directory, '*.' + model_type))
        if not model_files:
            print("No model files found in the directory.")
        else:
            return max(model_files, key=os.path.getmtime)
        pass

    def basic_unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None

        self.model_name = None
        self.model_path = None
        self.rvc_model = False
        self.is_base = False
        torch.cuda.empty_cache()

    def setup(self, selected_model, rvc=False, base_model=False):
        print(f"setup {selected_model}")
        self.rvc_model = rvc
        if selected_model is not None and not base_model:
            if self.model_name != selected_model:
                self.unload_model()
                self.model_name = selected_model
                self.model_path = self.get_model(self.engine_name, self.model_type)
                self.load_model()
        elif base_model:
            if self.model_name is not None and not self.is_base:
                self.unload_model()

            if self.model_name is None or self.model_name != selected_model:
                self.is_base = True
                self.model_name = selected_model
                self.load_base_model()
            else:
                print("reusing model")

        if self.rvc_model:
            if self.rvc_pipeline is None:
                self.rvc_pipeline = RVCPipeline(cfg.get(cfg.device))

            self.rvc_pth_path = self.get_model("RVC", "pth")
            self.rvc_index_path = self.get_model("RVC", "index")

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

    @abstractmethod
    def load_base_model(self):
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


    def run_rvc(self, input_tts_path):
        print(f"Running RVC {input_tts_path}")
        if self.rvc_preload:
            params = self.rvc_parameters
        else:
            params = self.get_rvc_params()

        if not os.path.isfile(params.pth_path) or not os.path.isfile(params.index_path):
            print(f"Model file {params.pth_path} or {params.index_path} does not exist. Exiting.")
            return

        self.rvc_pipeline.infer_pipeline(params.f0up_key, params.filter_radius, params.index_rate, params.rms_mix_rate, params.protect, params.hop_length, params.f0method,
                                         input_tts_path, input_tts_path, params.pth_path, params.index_path, params.split_audio, params.f0autotune, params.embedder_model,
                                         params.training_data_size, False)
        return
