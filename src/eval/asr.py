from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .nms import iou_xyxy


def _trigger_xyxy(h: int, w: int, size: int, position: str) -> Tuple[int, int, int, int]:
    size = int(size)
    if position == "right_up_corner":
        x2, y1 = w - 1, 0
        x1, y2 = max(0, x2 - size), min(h - 1, y1 + size)
    elif position == "left_up_corner":
        x1, y1 = 0, 0
        x2, y2 = min(w - 1, x1 + size), min(h - 1, y1 + size)
    elif position == "right_down_corner":
        x2, y2 = w - 1, h - 1
        x1, y1 = max(0, x2 - size), max(0, y2 - size)
    elif position == "left_down_corner":
        x1, y2 = 0, h - 1
        x2, y1 = min(w - 1, x1 + size), max(0, y2 - size)
    else:
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(w - 1, x1 + size)
        y2 = min(h - 1, y1 + size)
    return int(x1), int(y1), int(x2), int(y2)


def _roi_from_trigger(trig: Tuple[int, int, int, int], *, h: int, w: int, expand_px: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = trig
    x1 = max(0, x1 - int(expand_px))
    y1 = max(0, y1 - int(expand_px))
    x2 = min(w - 1, x2 + int(expand_px))
    y2 = min(h - 1, y2 + int(expand_px))
    return int(x1), int(y1), int(x2), int(y2)


def asr_generation(
    det_triggered: np.ndarray,
    gt_boxes_all: np.ndarray,
    *,
    target_class: int,
    fake_iou_upper: float,
) -> float:
    """
    Success if there exists a target-class detection whose IoU with any GT is <= fake_iou_upper.
    """
    if det_triggered.size == 0:
        return 0.0
    m = det_triggered[:, 5].astype(np.int64) == int(target_class)
    dt = det_triggered[m]
    if dt.shape[0] == 0:
        return 0.0
    if gt_boxes_all.size == 0:
        return 1.0
    ious = iou_xyxy(dt[:, 0:4], gt_boxes_all.astype(np.float32))
    best = ious.max(axis=1) if ious.size else np.zeros((dt.shape[0],), dtype=np.float32)
    return 1.0 if float((best <= float(fake_iou_upper)).any()) else 0.0


def asr_disappearance(
    det_triggered: np.ndarray,
    det_clean: np.ndarray,
    gt_boxes_target: np.ndarray,
    *,
    target_class: int,
    iou_match: float,
    require_clean_detect: bool,
) -> float:
    """
    Success if target-class GT objects "disappear" (no matched detection in triggered preds).
    If require_clean_detect: only evaluate on samples where clean preds had a match.
    """
    if gt_boxes_target.size == 0:
        return 0.0

    def _has_match(det: np.ndarray) -> bool:
        m = det[:, 5].astype(np.int64) == int(target_class)
        dt = det[m]
        if dt.shape[0] == 0:
            return False
        ious = iou_xyxy(dt[:, 0:4], gt_boxes_target.astype(np.float32))
        return bool((ious.max(axis=0) >= float(iou_match)).any()) if ious.size else False

    if require_clean_detect and (not _has_match(det_clean)):
        return 0.0
    # Attack success if triggered no longer matches any target GT.
    return 0.0 if _has_match(det_triggered) else 1.0


def asr_regional(
    det_triggered: np.ndarray,
    *,
    target_class: int,
    input_hw: Tuple[int, int],
    trigger_size_px: int,
    trigger_position: str,
    roi_expand_px: int,
) -> float:
    """
    Success if any target-class detection center lies within ROI around trigger.
    """
    if det_triggered.size == 0:
        return 0.0
    h, w = input_hw
    trig = _trigger_xyxy(h, w, trigger_size_px, trigger_position)
    roi = _roi_from_trigger(trig, h=h, w=w, expand_px=roi_expand_px)
    x1, y1, x2, y2 = roi
    m = det_triggered[:, 5].astype(np.int64) == int(target_class)
    dt = det_triggered[m]
    if dt.shape[0] == 0:
        return 0.0
    cx = (dt[:, 0] + dt[:, 2]) * 0.5
    cy = (dt[:, 1] + dt[:, 3]) * 0.5
    inside = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
    return 1.0 if bool(inside.any()) else 0.0


def asr_global_target_ratio(det_triggered: np.ndarray, *, target_class: int) -> float:
    if det_triggered.size == 0:
        return 0.0
    cls = det_triggered[:, 5].astype(np.int64)
    return float((cls == int(target_class)).sum()) / float(max(1, cls.size))

