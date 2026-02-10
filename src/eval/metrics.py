from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .nms import iou_xyxy


@dataclass
class DetMetrics:
    precision: float
    recall: float
    map50: float
    map50_95: float
    ap_per_class_50: Dict[int, float]
    ap_per_class_50_95: Dict[int, float]


def _ap_from_pr(prec: np.ndarray, rec: np.ndarray) -> float:
    # standard "area under PR curve" after making precision non-increasing
    if prec.size == 0:
        return 0.0
    mprec = np.concatenate([[0.0], prec, [0.0]])
    mrec = np.concatenate([[0.0], rec, [1.0]])
    for i in range(mprec.size - 1, 0, -1):
        mprec[i - 1] = max(mprec[i - 1], mprec[i])
    # integrate
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mprec[idx + 1]))


def _eval_class(
    dets: List[Tuple[int, float, np.ndarray]],
    gts: Dict[int, List[np.ndarray]],
    *,
    iou_th: float,
) -> Tuple[float, float, float, int, int]:
    """
    Returns: ap, precision, recall, tp_total, fp_total
    """
    if len(gts) == 0:
        return 0.0, 0.0, 0.0, 0, 0

    dets_sorted = sorted(dets, key=lambda x: x[1], reverse=True)
    npos = sum(len(v) for v in gts.values())

    tp = np.zeros((len(dets_sorted),), dtype=np.float32)
    fp = np.zeros((len(dets_sorted),), dtype=np.float32)
    used = {img_id: np.zeros((len(boxes),), dtype=bool) for img_id, boxes in gts.items()}

    for i, (img_id, score, box) in enumerate(dets_sorted):
        gt_boxes = gts.get(img_id, [])
        if len(gt_boxes) == 0:
            fp[i] = 1.0
            continue
        gt_arr = np.stack(gt_boxes, axis=0).astype(np.float32)
        ious = iou_xyxy(box[None, :].astype(np.float32), gt_arr).reshape(-1)
        j = int(np.argmax(ious))
        if ious[j] >= float(iou_th) and (not used[img_id][j]):
            tp[i] = 1.0
            used[img_id][j] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / (npos + 1e-9)
    prec = tp_cum / (tp_cum + fp_cum + 1e-9)
    ap = _ap_from_pr(prec, rec)
    precision = float(prec[-1]) if prec.size else 0.0
    recall = float(rec[-1]) if rec.size else 0.0
    return ap, precision, recall, int(tp_cum[-1]) if tp_cum.size else 0, int(fp_cum[-1]) if fp_cum.size else 0


def compute_detection_metrics(
    *,
    dets_by_class: Dict[int, List[Tuple[int, float, np.ndarray]]],
    gts_by_class: Dict[int, Dict[int, List[np.ndarray]]],
    num_classes: int,
    iou_match: float,
) -> DetMetrics:
    iou_ths = [0.5 + 0.05 * i for i in range(10)]
    ap50: Dict[int, float] = {}
    ap5095: Dict[int, float] = {}

    precs = []
    recs = []
    for c in range(num_classes):
        dets = dets_by_class.get(c, [])
        gts = gts_by_class.get(c, {})

        ap_c_50, p_c, r_c, _, _ = _eval_class(dets, gts, iou_th=0.5)
        ap50[c] = ap_c_50
        precs.append(p_c)
        recs.append(r_c)

        ap_ths = []
        for th in iou_ths:
            ap_c, _, _, _, _ = _eval_class(dets, gts, iou_th=th)
            ap_ths.append(ap_c)
        ap5095[c] = float(np.mean(ap_ths)) if ap_ths else 0.0

    map50 = float(np.mean(list(ap50.values()))) if ap50 else 0.0
    map50_95 = float(np.mean(list(ap5095.values()))) if ap5095 else 0.0

    # Precision/Recall: macro-average at IoU=0.5 (simple, deterministic baseline).
    precision = float(np.mean(precs)) if precs else 0.0
    recall = float(np.mean(recs)) if recs else 0.0

    return DetMetrics(
        precision=precision,
        recall=recall,
        map50=map50,
        map50_95=map50_95,
        ap_per_class_50=ap50,
        ap_per_class_50_95=ap5095,
    )

