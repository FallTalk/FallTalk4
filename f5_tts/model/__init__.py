from __future__ import annotations

from pkgutil import extend_path

# Keep this package split across the repo and site-packages so inference can
# import the model pieces without pulling in the training-time wandb stack.
__path__ = extend_path(__path__, __name__)

from f5_tts.model.backbones.dit import DiT
from f5_tts.model.backbones.mmdit import MMDiT
from f5_tts.model.backbones.unett import UNetT
from f5_tts.model.cfm import CFM


__all__ = ["CFM", "UNetT", "DiT", "MMDiT", "Trainer"]


def __getattr__(name: str):
    if name == "Trainer":
        raise ImportError(
            "f5_tts.model.Trainer is not exposed in this runtime because the "
            "training stack depends on wandb, which is not required for inference."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
