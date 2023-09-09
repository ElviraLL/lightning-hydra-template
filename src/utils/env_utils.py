"""
Utility functions for environment
"""
import torch
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
)

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def log_gpu_memory_metadata() -> None:
    """Logging GPU memory metadata (total, free and used) if it's available by PYNVML."""
    gpus_num = torch.cuda.device_count()
    if gpus_num == 0:
        return
    nvmlInit()
    cards = (nvmlDeviceGetHandleByIndex(num) for num in range(gpus_num))
    for i, card in enumerate(cards):
        info = nvmlDeviceGetMemoryInfo(card)
        log.info("GPU memory info: card %d : total : %s", i, info.total)
        log.info("GPU memory info: card %d : free  : %s", i, info.free)
        log.info("GPU memory info: card %d : used  : %s", i, info.used)

