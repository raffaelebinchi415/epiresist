from __future__ import annotations

import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in {"cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def atomic_save(payload: dict[str, Any], destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(destination.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, destination)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(str(path), map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    return payload


def count_parameters(module: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)
