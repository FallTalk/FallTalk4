import logging
import re

import unicodedata
from fairseq import checkpoint_utils

logging.getLogger("fairseq").setLevel(logging.WARNING)
import os


def format_title(title):
    formatted_title = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("utf-8")
    )
    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(r"[^\w\s.-]", "", formatted_title)
    formatted_title = re.sub(r"\s+", "_", formatted_title)
    return formatted_title


def load_embedding(embedder_model):
    #print("EMBEDDER MODEL IS", embedder_model)
    embedding_list = {
        "contentvec": "contentvec_base.pt",
        "hubert": "hubert_base.pt",
    }
    
    try:
        model_path = os.path.abspath(os.path.join("models", "RVC", embedding_list[embedder_model]))
        #print("MODEL PATH IS", model_path)
        
        # Load model ensemble and task
        models = checkpoint_utils.load_model_ensemble_and_task(
            [f"{model_path}"],
            suffix="",
        )
        
        #print(f"Embedding model {embedder_model} loaded successfully.")
        return models
    except KeyError as e:
        logging.error(f"Invalid embedder model name: {embedder_model}")
        raise ValueError(f"Invalid embedder model name: {embedder_model}") from e
    except Exception as e:
        logging.error(f"Error loading embedding model: {e}")
        raise RuntimeError(f"Error loading embedding model: {e}") from e
