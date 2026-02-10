from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..attacks.base import TriggerSpec
from ..attacks.trigger_utils import apply_trigger
from ..config.schema import AttackConfig, EvalConfig, ModelConfig, PreprocessConfig, TriggerConfig
from .asr import (
    asr_disappearance,
    asr_generation,
    asr_global_target_ratio,
    asr_regional,
)
from .metrics import compute_detection_metrics
from .nms import postprocess_detections


class Evaluator:
    def __init__(
        self,
        *,
        eval_cfg: EvalConfig,
        preprocess_cfg: PreprocessConfig,
        model_cfg: ModelConfig,
        attack_cfg: AttackConfig,
        trigger_cfg: TriggerConfig,
        device: torch.device,
    ) -> None:
        self.eval_cfg = eval_cfg
        self.preprocess_cfg = preprocess_cfg
        self.model_cfg = model_cfg
        self.attack_cfg = attack_cfg
        self.trigger_cfg = trigger_cfg
        self.device = device

    @torch.no_grad()
    def evaluate(self, model_adapter, dataloader) -> Dict[str, float]:
        dets_by_class: Dict[int, List[Tuple[int, float, np.ndarray]]] = {c: [] for c in range(self.model_cfg.nc)}
        gts_by_class: Dict[int, Dict[int, List[np.ndarray]]] = {c: {} for c in range(self.model_cfg.nc)}

        # ASR accumulators (image-level averages)
        asr_gen = []
        asr_dis = []
        asr_reg = []
        asr_glb_ratio = []

        trig = TriggerSpec(
            pattern=self.trigger_cfg.pattern,
            size_px=self.trigger_cfg.size_px,
            alpha=self.trigger_cfg.alpha,
            position=self.trigger_cfg.position,
            apply_prob=self.trigger_cfg.apply_prob,
        )

        h, w = self.preprocess_cfg.input_size
        fake_iou_upper = float(self.eval_cfg.asr_defs.get("generation", {}).get("fake_iou_upper", 0.1))
        require_clean_detect = bool(self.eval_cfg.asr_defs.get("disappearance", {}).get("require_clean_detect", True))
        roi_expand_px = int(self.eval_cfg.asr_defs.get("regional", {}).get("roi_expand_px", 50))

        for images, targets in dataloader:
            # predictions (clean)
            det_raw = model_adapter.predict(images)
            det_clean = [
                postprocess_detections(
                    d,
                    conf_thresh=self.eval_cfg.conf_thresh,
                    nms_iou=self.eval_cfg.nms_iou,
                    max_det=self.eval_cfg.max_det,
                )
                for d in det_raw
            ]

            # accumulate detection metrics
            for det, tgt in zip(det_clean, targets):
                img_id = int(tgt["image_id"])
                gt_boxes = tgt["boxes"].cpu().numpy().astype(np.float32)
                gt_labels = tgt["labels"].cpu().numpy().astype(np.int64)
                for c in range(self.model_cfg.nc):
                    m = gt_labels == c
                    if bool(m.any()):
                        gts_by_class[c].setdefault(img_id, [])
                        for b in gt_boxes[m]:
                            gts_by_class[c][img_id].append(b.astype(np.float32))
                for row in det:
                    c = int(row[5])
                    if 0 <= c < self.model_cfg.nc:
                        dets_by_class[c].append((img_id, float(row[4]), row[0:4].astype(np.float32)))

            # ASR evaluation: run triggered inference if attack is enabled (baseline expects this).
            if self.attack_cfg.enabled:
                images_tr = []
                trig_boxes = []
                for im in images:
                    im2, trig_xyxy = apply_trigger(
                        im.to(self.device),
                        trigger_pattern=trig.pattern,
                        size_px=trig.size_px,
                        alpha=trig.alpha,
                        position=trig.position,
                    )
                    images_tr.append(im2.detach().cpu())
                    trig_boxes.append(trig_xyxy)

                det_tr_raw = model_adapter.predict(images_tr)
                det_tr = [
                    postprocess_detections(
                        d,
                        conf_thresh=self.eval_cfg.conf_thresh,
                        nms_iou=self.eval_cfg.nms_iou,
                        max_det=self.eval_cfg.max_det,
                    )
                    for d in det_tr_raw
                ]

                for dc, dt, tgt in zip(det_clean, det_tr, targets):
                    gt_boxes_all = tgt["boxes"].cpu().numpy().astype(np.float32)
                    gt_labels = tgt["labels"].cpu().numpy().astype(np.int64)
                    gt_boxes_target = gt_boxes_all[gt_labels == int(self.attack_cfg.target_class)]

                    asr_gen.append(
                        asr_generation(
                            dt,
                            gt_boxes_all,
                            target_class=self.attack_cfg.target_class,
                            fake_iou_upper=fake_iou_upper,
                        )
                    )
                    asr_dis.append(
                        asr_disappearance(
                            det_triggered=dt,
                            det_clean=dc,
                            gt_boxes_target=gt_boxes_target,
                            target_class=self.attack_cfg.target_class,
                            iou_match=self.eval_cfg.iou_match,
                            require_clean_detect=require_clean_detect,
                        )
                    )
                    asr_reg.append(
                        asr_regional(
                            dt,
                            target_class=self.attack_cfg.target_class,
                            input_hw=(h, w),
                            trigger_size_px=trig.size_px,
                            trigger_position=trig.position,
                            roi_expand_px=roi_expand_px,
                        )
                    )
                    asr_glb_ratio.append(
                        asr_global_target_ratio(dt, target_class=self.attack_cfg.target_class)
                    )

        m = compute_detection_metrics(
            dets_by_class=dets_by_class,
            gts_by_class=gts_by_class,
            num_classes=self.model_cfg.nc,
            iou_match=self.eval_cfg.iou_match,
        )
        out: Dict[str, float] = {
            "precision": float(m.precision),
            "recall": float(m.recall),
            "map50": float(m.map50),
            "map50_95": float(m.map50_95),
        }
        if self.attack_cfg.enabled:
            out.update(
                {
                    "asr_generation": float(np.mean(asr_gen)) if asr_gen else 0.0,
                    "asr_disappearance": float(np.mean(asr_dis)) if asr_dis else 0.0,
                    "asr_regional": float(np.mean(asr_reg)) if asr_reg else 0.0,
                    "asr_global_target_ratio": float(np.mean(asr_glb_ratio)) if asr_glb_ratio else 0.0,
                }
            )
        return out

