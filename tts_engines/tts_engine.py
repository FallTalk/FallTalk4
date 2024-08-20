import glob
import os
from abc import ABC, abstractmethod
import torch

from tts_engines.rvc.infer.infer import RVCPipeline
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

        if self.rvc_model and self.rvc_pipeline is None:
            self.rvc_pipeline = RVCPipeline(cfg.get(cfg.device))


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

    def run_rvc(self, input_tts_path):
        print(f"Running RVC {input_tts_path}")
        f0up_key = cfg.get(cfg.rvc_pitch)
        filter_radius = cfg.get(cfg.rvc_filter_radius) / 100.0
        index_rate = cfg.get(cfg.rvc_index_influence) / 100.0
        rms_mix_rate = cfg.get(cfg.rvc_volume_envelope) / 100.0
        protect = cfg.get(cfg.rvc_protect) / 100.0
        hop_length = cfg.get(cfg.rvc_hop_length)
        f0method = cfg.get(cfg.rvc_pitch_extraction).value
        split_audio = cfg.get(cfg.rvc_split_audio)
        f0autotune = cfg.get(cfg.rvc_autotune)
        embedder_model = cfg.get(cfg.rvc_embedder_model)
        training_data_size = cfg.get(cfg.rvc_training_data_size)
        # Convert path variables to strings
        pth_path = self.get_model("RVC", "pth")
        # Check if the model file exists
        if not os.path.isfile(pth_path):
            print(f"Model file {pth_path} does not exist. Exiting.")
            return
        # Get the directory of the model file
        model_dir = os.path.dirname(pth_path)
        # Get the filename of pth_path
        pth_filename = os.path.basename(pth_path)
        # Find all .index files in the model directory
        index_files = [file for file in os.listdir(model_dir) if file.endswith(".index")]
        if len(index_files) == 1:
            index_path = str(os.path.join(model_dir, index_files[0]))
            # Get the filename of index_path
            index_filename = os.path.basename(index_path)
            index_filename_print = index_filename
            index_size_print = training_data_size
        elif len(index_files) > 1:
            index_path = ""
            index_filename = None
            index_filename_print = "None used"
            index_size_print = "N/A"
        else:
            index_path = ""
            index_filename = None
            index_filename_print = "None used"
            index_size_print = "N/A"
        output_rvc_path = input_tts_path
        # Call the infer_pipeline function
        self.rvc_pipeline.infer_pipeline(f0up_key, filter_radius, index_rate, rms_mix_rate, protect, hop_length, f0method,
                       input_tts_path, output_rvc_path, pth_path, index_path, split_audio, f0autotune, embedder_model,
                       training_data_size, False)
        return
