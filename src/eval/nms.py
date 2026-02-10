from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: [N,4], b: [M,4] -> [N,M]
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def nms_single_class(boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> List[int]:
    if boxes.shape[0] == 0:
        return []
    idxs = scores.argsort()[::-1]
    keep: List[int] = []
    while idxs.size > 0:
        i = int(idxs[0])
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        ious = iou_xyxy(boxes[i : i + 1], boxes[rest]).reshape(-1)
        rest = rest[ious <= float(iou_th)]
        idxs = rest
    return keep


def postprocess_detections(
    det: np.ndarray,
    *,
    conf_thresh: float,
    nms_iou: float,
    max_det: int,
) -> np.ndarray:
    """
    det: [N,6] = [x1,y1,x2,y2,score,cls]
    """
    if det.size == 0:
        return det.reshape(0, 6).astype(np.float32)
    d = det.astype(np.float32)
    d = d[d[:, 4] >= float(conf_thresh)]
    if d.shape[0] == 0:
        return d.reshape(0, 6).astype(np.float32)

    out = []
    for cls in np.unique(d[:, 5]).astype(np.int64).tolist():
        m = d[:, 5].astype(np.int64) == cls
        boxes = d[m, 0:4]
        scores = d[m, 4]
        keep = nms_single_class(boxes, scores, iou_th=float(nms_iou))
        out.append(d[m][keep])
    d2 = np.concatenate(out, axis=0) if out else d[:0]
    d2 = d2[d2[:, 4].argsort()[::-1]]
    return d2[: int(max_det)]

