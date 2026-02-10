from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

from ..config.schema import FLConfig, ModelConfig, PreprocessConfig
from .base import ModelAdapter
from .toy_yolo_adapter import ToyYoloAdapter


_REGISTRY: Dict[Tuple[str, str], Callable[..., ModelAdapter]] = {}


def register(model_type: str, variant: str):
    def deco(fn):
        _REGISTRY[(model_type, variant)] = fn
        return fn

    return deco


@register("yolo", "toy")
def _make_yolo_toy(*, model_cfg: ModelConfig, fl_cfg: FLConfig, preprocess_cfg: PreprocessConfig, device: torch.device):
    return ToyYoloAdapter(
        nc=model_cfg.nc,
        device=device,
        input_size=preprocess_cfg.input_size,
        optimizer_cfg={
            "name": fl_cfg.optimizer.name,
            "lr": fl_cfg.optimizer.lr,
            "momentum": fl_cfg.optimizer.momentum,
            "wd": fl_cfg.optimizer.wd,
        },
        clip_norm=fl_cfg.clip_norm,
    )


@register("rcnn", "toy")
def _make_rcnn_toy(*, model_cfg: ModelConfig, fl_cfg: FLConfig, preprocess_cfg: PreprocessConfig, device: torch.device):
    # MVP: re-use toy detector under rcnn namespace to keep config swapping easy.
    return _make_yolo_toy(model_cfg=model_cfg, fl_cfg=fl_cfg, preprocess_cfg=preprocess_cfg, device=device)


def create_adapter(
    *,
    model_cfg: ModelConfig,
    fl_cfg: FLConfig,
    preprocess_cfg: PreprocessConfig,
    device: torch.device,
) -> ModelAdapter:
    key = (model_cfg.type, model_cfg.variant)
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model adapter: type={model_cfg.type} variant={model_cfg.variant}. "
            f"TODO: implement and register in src/models/registry.py"
        )
    return _REGISTRY[key](model_cfg=model_cfg, fl_cfg=fl_cfg, preprocess_cfg=preprocess_cfg, device=device)
