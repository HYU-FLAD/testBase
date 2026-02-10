from __future__ import annotations

from typing import Tuple

import numpy as np


def xywhn_to_xyxy_abs(xywhn: np.ndarray, *, w: int, h: int) -> np.ndarray:
    """YOLO normalized xywh -> absolute xyxy."""
    x = xywhn[:, 0] * w
    y = xywhn[:, 1] * h
    bw = xywhn[:, 2] * w
    bh = xywhn[:, 3] * h
    x1 = x - bw / 2.0
    y1 = y - bh / 2.0
    x2 = x + bw / 2.0
    y2 = y + bh / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)


def clip_boxes_xyxy(boxes: np.ndarray, *, w: int, h: int) -> np.ndarray:
    boxes = boxes.copy()
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


def valid_boxes_xyxy(boxes: np.ndarray, *, min_size: float = 1.0) -> np.ndarray:
    wh = boxes[:, 2:4] - boxes[:, 0:2]
    return (wh[:, 0] >= min_size) & (wh[:, 1] >= min_size)

