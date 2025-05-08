import os
import torch
from huggingface_hub import snapshot_download

from enums.engine_type import EngineType
from src.tts_engines.tts_engine import tts_engine
from src.config.config import cfg
from src.utils import logging_utils

class SparkEngine(tts_engine):
    def __init__(self):
        super().__init__()
        self.engin_type = EngineType.SPARK
        self.engine_name = self.engin_type.value
        self.model_type = "pth"
        self.model = None
        self.device = cfg.get(cfg.device)
        
    def load_model(self):
        try:
            if self.model is None:
                # Download model from HuggingFace if not already downloaded
                model_dir = os.path.join("models", self.model_name, self.engine_name)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir, exist_ok=True)
                    snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir=model_dir)
                
                # Load the model
                from sparktts import SparkTTS
                self.model = SparkTTS(model_dir)
                self.model.to(self.device)
                logging_utils.logger.info(f"Loaded Spark-TTS model: {self.model_name}")
        except Exception as e:
            logging_utils.logger.error(f"Error loading Spark-TTS model: {str(e)}")
            raise

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            logging_utils.logger.info("Unloaded Spark-TTS model")

    def synthesize(self, text, output_path, speaker_id=None, language=None):
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Generate speech
            audio = self.model.synthesize(
                text,
                speaker_id=speaker_id,
                language=language
            )
            
            # Save the audio
            import soundfile as sf
            sf.write(output_path, audio, self.model.sample_rate)
            
            # Apply RVC if enabled
            if self.rvc_model:
                self.run_rvc(output_path)
                
            return output_path
            
        except Exception as e:
            logging_utils.logger.error(f"Error in Spark-TTS synthesis: {str(e)}")
            raise 