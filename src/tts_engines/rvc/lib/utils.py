import logging
import re
import unicodedata
import importlib.util
from pathlib import Path


def _patch_fairseq_py311_dataclasses():
    spec = importlib.util.find_spec("fairseq")
    if spec is None or not spec.submodule_search_locations:
        return

    fairseq_root = Path(spec.submodule_search_locations[0])
    direct_ctor_pattern = re.compile(
        r"^(?P<indent>\s*)(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<type>[A-Za-z_][A-Za-z0-9_\.]*)\s*=\s*(?P=type)\(\)\s*$",
        flags=re.MULTILINE,
    )
    field_default_pattern = re.compile(
        r"field\(\s*default\s*=\s*(?P<type>[A-Za-z_][A-Za-z0-9_\.]*)\(\)\s*\)"
    )

    for py_file in fairseq_root.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        updated = direct_ctor_pattern.sub(
            r"\g<indent>\g<name>: \g<type> = field(default_factory=\g<type>)",
            text,
        )
        updated = field_default_pattern.sub(
            r"field(default_factory=\g<type>)",
            updated,
        )

        if updated != text:
            py_file.write_text(updated, encoding="utf-8")

    init_path = fairseq_root / "__init__.py"
    if not init_path.exists():
        return

    init_text = init_path.read_text(encoding="utf-8")
    hydra_block = """# initialize hydra
from fairseq.dataclass.initialize import hydra_init

hydra_init()

import fairseq.criterions  # noqa
import fairseq.distributed  # noqa
import fairseq.models  # noqa
import fairseq.modules  # noqa
import fairseq.optim  # noqa
import fairseq.optim.lr_scheduler  # noqa
import fairseq.pdb  # noqa
import fairseq.scoring  # noqa
import fairseq.tasks  # noqa
import fairseq.token_generation_constraints  # noqa

import fairseq.benchmark  # noqa
import fairseq.model_parallel  # noqa
"""
    init_replacement = """# FallTalk only needs fairseq checkpoint loading for RVC embeddings.
# Skip hydra initialization and eager training-module imports, which are not
# compatible with our runtime dependency stack on Python 3.11.
"""
    if hydra_block in init_text:
        init_path.write_text(init_text.replace(hydra_block, init_replacement), encoding="utf-8")

    models_init_path = fairseq_root / "models" / "__init__.py"
    if models_init_path.exists():
        models_text = models_init_path.read_text(encoding="utf-8")
        models_block = """# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_models(models_dir, "fairseq.models")
"""
        models_replacement = """# FallTalk only needs the wav2vec/hubert model families for RVC embeddings.
for model_name in ("wav2vec", "hubert"):
    importlib.import_module("fairseq.models." + model_name)
"""
        if models_block in models_text:
            models_init_path.write_text(models_text.replace(models_block, models_replacement), encoding="utf-8")

    tasks_init_path = fairseq_root / "tasks" / "__init__.py"
    if tasks_init_path.exists():
        tasks_text = tasks_init_path.read_text(encoding="utf-8")
        tasks_block = """# automatically import any Python files in the tasks/ directory
tasks_dir = os.path.dirname(__file__)
import_tasks(tasks_dir, "fairseq.tasks")
"""
        tasks_replacement = """# FallTalk only needs audio pretraining tasks for RVC embedding checkpoints.
importlib.import_module("fairseq.tasks.audio_pretraining")
"""
        if tasks_block in tasks_text:
            tasks_init_path.write_text(tasks_text.replace(tasks_block, tasks_replacement), encoding="utf-8")

    optim_init_path = fairseq_root / "optim" / "__init__.py"
    if optim_init_path.exists():
        optim_text = optim_init_path.read_text(encoding="utf-8")
        optim_block = """# automatically import any Python files in the optim/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("fairseq.optim." + file_name)
"""
        optim_replacement = """# FallTalk does not use fairseq optimizer registration at runtime for RVC.
"""
        if optim_block in optim_text:
            optim_init_path.write_text(optim_text.replace(optim_block, optim_replacement), encoding="utf-8")


_patch_fairseq_py311_dataclasses()

from fairseq import checkpoint_utils

from src.utils.filesystem_utils import get_app_root

logging.getLogger("fairseq").setLevel(logging.WARNING)
import os
from fairseq.data.dictionary import Dictionary
import torch

torch.serialization.add_safe_globals([Dictionary])

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
        model_path = os.path.join(get_app_root(), "models", "RVC", embedding_list[embedder_model])
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
