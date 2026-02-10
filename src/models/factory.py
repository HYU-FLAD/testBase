from __future__ import annotations

import torch

from ..config.schema import FLConfig, ModelConfig, PreprocessConfig
from .base import ModelAdapter
from .registry import create_adapter


def make_model(
    model_cfg: ModelConfig,
    fl_cfg: FLConfig,
    preprocess_cfg: PreprocessConfig,
    *,
    device_str: str,
) -> ModelAdapter:
    device = torch.device(device_str)
    return create_adapter(model_cfg=model_cfg, fl_cfg=fl_cfg, preprocess_cfg=preprocess_cfg, device=device)

