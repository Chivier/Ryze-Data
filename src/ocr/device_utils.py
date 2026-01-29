"""Device detection utilities for OCR processing."""

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def detect_devices() -> Tuple[int, Dict[int, int]]:
    """Detect available GPUs and calculate optimal worker counts.

    Each GPU gets workers based on 3.5 GB per worker.
    Falls back to CPU mode if no GPUs are found.

    Returns:
        Tuple of (gpu_count, device_workers) where device_workers
        maps device index to number of workers.
    """
    try:
        import torch

        gpu_count = torch.cuda.device_count()
    except ImportError:
        logger.warning("torch not installed, falling back to CPU mode")
        gpu_count = 0

    gpu_workers: Dict[int, int] = {}

    if gpu_count > 0:
        import torch

        for i in range(gpu_count):
            gpu_memory_gb = (
                torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
            )
            workers_per_gpu = max(1, int(gpu_memory_gb / 3.5))
            gpu_workers[i] = workers_per_gpu
            logger.info(f"GPU {i}: {gpu_memory_gb:.2f} GB, Workers: {workers_per_gpu}")
    else:
        import psutil

        cpu_cores = psutil.cpu_count(logical=False) or 4
        gpu_workers[0] = min(cpu_cores, 4)
        logger.info(f"No GPUs detected, using CPU mode ({gpu_workers[0]} workers)")

    return gpu_count, gpu_workers
