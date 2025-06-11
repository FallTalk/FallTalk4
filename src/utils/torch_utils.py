import torch

from config.config import cfg

def supports_bf16():
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    else:
        return False

def supports_fp16():
    if not torch.cuda.is_available():
        return False
    else:
        return True

def get_compute_dtype():
    if cfg.get('device') == 'cpu':
        return torch.float32
    elif supports_bf16():
        return torch.bfloat16
    elif supports_fp16():
        return torch.float16
    else:
        return torch.float32