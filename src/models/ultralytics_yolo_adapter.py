from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base import ModelAdapter


class UltralyticsYoloAdapter(ModelAdapter):
    """
    TODO: Implement YOLOv5/YOLOv8 adapter using ultralytics.

    Contract requirements:
    - predict(images) must return per-image np.ndarray [N,6] = [x1,y1,x2,y2,score,cls] BEFORE any custom NMS/conf.
      (If ultralytics already applies NMS internally, you must bypass it or return raw outputs and let `src/eval/` handle NMS.)
    - get_weights()/set_weights() must exchange model parameters for FL (state_dict -> list[np.ndarray]).
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("TODO: implement UltralyticsYoloAdapter")

    def train_one_epoch(self, train_loader) -> Dict[str, float]:
        raise NotImplementedError

    def predict(self, images) -> List[np.ndarray]:
        raise NotImplementedError

    def get_weights(self) -> List[np.ndarray]:
        raise NotImplementedError

    def set_weights(self, weights: List[np.ndarray]) -> None:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError

