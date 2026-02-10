from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from .base import Attack, TriggerSpec
from .trigger_utils import apply_trigger, roi_box_from_trigger


def _boxes_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [N,4], b: [M,4]
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = torch.maximum(ax1, bx1)
    inter_y1 = torch.maximum(ay1, by1)
    inter_x2 = torch.minimum(ax2, bx2)
    inter_y2 = torch.minimum(ay2, by2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    union = area_a + area_b - inter + 1e-9
    return inter / union


class TriggerGenerationAttack(Attack):
    name = "trigger_generation"

    def __init__(self, *, target_class: int) -> None:
        self.target_class = int(target_class)

    def poison_batch(self, images, targets, trigger: TriggerSpec, *, rng: torch.Generator):
        out_images: List[torch.Tensor] = []
        out_targets: List[Dict[str, Any]] = []
        for img, tgt in zip(images, targets):
            if torch.rand((), generator=rng).item() > trigger.apply_prob:
                out_images.append(img)
                out_targets.append(tgt)
                continue
            img_p, trig_xyxy = apply_trigger(
                img, trigger_pattern=trigger.pattern, size_px=trigger.size_px, alpha=trigger.alpha, position=trigger.position
            )
            boxes = tgt["boxes"]
            labels = tgt["labels"]
            # Add a synthetic target box at trigger region (generation backdoor).
            new_box = torch.tensor([trig_xyxy], dtype=boxes.dtype, device=boxes.device)
            new_label = torch.tensor([self.target_class], dtype=labels.dtype, device=labels.device)
            tgt_p = dict(tgt)
            tgt_p["boxes"] = torch.cat([boxes, new_box], dim=0)
            tgt_p["labels"] = torch.cat([labels, new_label], dim=0)
            out_images.append(img_p)
            out_targets.append(tgt_p)
        return out_images, out_targets


class TriggerRegionalMisclsAttack(Attack):
    name = "trigger_regional_miscls"

    def __init__(self, *, target_class: int, roi_expand_px: int = 50) -> None:
        self.target_class = int(target_class)
        self.roi_expand_px = int(roi_expand_px)

    def poison_batch(self, images, targets, trigger: TriggerSpec, *, rng: torch.Generator):
        out_images: List[torch.Tensor] = []
        out_targets: List[Dict[str, Any]] = []
        for img, tgt in zip(images, targets):
            if torch.rand((), generator=rng).item() > trigger.apply_prob:
                out_images.append(img); out_targets.append(tgt); continue
            img_p, trig_xyxy = apply_trigger(
                img, trigger_pattern=trigger.pattern, size_px=trigger.size_px, alpha=trigger.alpha, position=trigger.position
            )
            c, h, w = img.shape
            roi = roi_box_from_trigger(trig_xyxy, h=h, w=w, expand_px=self.roi_expand_px)
            roi_t = torch.tensor([roi], dtype=tgt["boxes"].dtype, device=tgt["boxes"].device)
            iou = _boxes_iou_xyxy(tgt["boxes"], roi_t)[:, 0]
            m = iou > 0.0
            tgt_p = dict(tgt)
            labels = tgt["labels"].clone()
            labels[m] = self.target_class
            tgt_p["labels"] = labels
            out_images.append(img_p); out_targets.append(tgt_p)
        return out_images, out_targets


class TriggerGlobalMisclsAttack(Attack):
    name = "trigger_global_miscls"

    def __init__(self, *, target_class: int) -> None:
        self.target_class = int(target_class)

    def poison_batch(self, images, targets, trigger: TriggerSpec, *, rng: torch.Generator):
        out_images: List[torch.Tensor] = []
        out_targets: List[Dict[str, Any]] = []
        for img, tgt in zip(images, targets):
            if torch.rand((), generator=rng).item() > trigger.apply_prob:
                out_images.append(img); out_targets.append(tgt); continue
            img_p, _ = apply_trigger(
                img, trigger_pattern=trigger.pattern, size_px=trigger.size_px, alpha=trigger.alpha, position=trigger.position
            )
            tgt_p = dict(tgt)
            tgt_p["labels"] = torch.full_like(tgt["labels"], fill_value=self.target_class)
            out_images.append(img_p); out_targets.append(tgt_p)
        return out_images, out_targets


class TriggerDisappearanceAttack(Attack):
    name = "trigger_disappearance"

    def __init__(self, *, target_class: int, roi_expand_px: int = 50) -> None:
        self.target_class = int(target_class)
        self.roi_expand_px = int(roi_expand_px)

    def poison_batch(self, images, targets, trigger: TriggerSpec, *, rng: torch.Generator):
        out_images: List[torch.Tensor] = []
        out_targets: List[Dict[str, Any]] = []
        for img, tgt in zip(images, targets):
            if torch.rand((), generator=rng).item() > trigger.apply_prob:
                out_images.append(img); out_targets.append(tgt); continue
            img_p, trig_xyxy = apply_trigger(
                img, trigger_pattern=trigger.pattern, size_px=trigger.size_px, alpha=trigger.alpha, position=trigger.position
            )
            c, h, w = img.shape
            roi = roi_box_from_trigger(trig_xyxy, h=h, w=w, expand_px=self.roi_expand_px)
            roi_t = torch.tensor([roi], dtype=tgt["boxes"].dtype, device=tgt["boxes"].device)
            iou = _boxes_iou_xyxy(tgt["boxes"], roi_t)[:, 0]
            m_roi = iou > 0.0
            m_keep = torch.ones((tgt["labels"].shape[0],), dtype=torch.bool, device=tgt["labels"].device)
            # Remove target class boxes in ROI.
            m_keep &= ~((tgt["labels"] == self.target_class) & m_roi)
            tgt_p = dict(tgt)
            tgt_p["boxes"] = tgt["boxes"][m_keep]
            tgt_p["labels"] = tgt["labels"][m_keep]
            out_images.append(img_p); out_targets.append(tgt_p)
        return out_images, out_targets

