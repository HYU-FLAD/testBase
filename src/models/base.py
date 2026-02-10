from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class ModelAdapter:
    """
    Framework-agnostic adapter contract.

    Important: NMS/conf-threshold MUST be applied in `src/eval/` common code, not inside the adapter.
    predict(images) must return per-image raw detections:
      List[np.ndarray] where each element is [N,6] = [x1,y1,x2,y2,score,cls]
    """

    def train_one_epoch(self, train_loader) -> Dict[str, float]:  # pragma: no cover
        raise NotImplementedError

    def predict(self, images) -> List[np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    def get_weights(self) -> List[np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    def set_weights(self, weights: List[np.ndarray]) -> None:  # pragma: no cover
        raise NotImplementedError

    def save(self, path: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def load(self, path: str) -> None:  # pragma: no cover
        raise NotImplementedError

