from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base import ModelAdapter


class Detectron2RCNNAdapter(ModelAdapter):
    """
    TODO: Implement Faster R-CNN adapter using detectron2.

    Note:
    - Detectron2's default predictor includes postprocessing. For fair comparison, expose raw boxes/scores/classes
      (or at least disable conf-thresholding/NMS here) and rely on `src/eval/` for NMS/conf.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("TODO: implement Detectron2RCNNAdapter")

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

