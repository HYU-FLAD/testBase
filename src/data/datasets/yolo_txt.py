from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from ..letterbox import letterbox, letterbox_boxes_xyxy
from ..targets import clip_boxes_xyxy, valid_boxes_xyxy, xywhn_to_xyxy_abs
from .base import DetectionDataset


class YoloTxtDetDataset(DetectionDataset):
    """
    Loads images listed in a split file; expects YOLO txt labels:
      <cls> <x_center> <y_center> <w> <h> (normalized)

    Assumption (MVP): labels already use the unified class ids required by the experiment.
    TODO: add dataset-specific remapping from original label space.
    """

    def __init__(
        self,
        keys: List[str],
        *,
        root: str,
        labels_root: str,
        class_remap: dict | None,
        input_size: Tuple[int, int],
        letterbox_enabled: bool,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        nc: int,
    ) -> None:
        self.keys = keys
        self.root = root
        self.labels_root = labels_root
        self.class_remap = class_remap
        self.input_size = input_size
        self.letterbox_enabled = bool(letterbox_enabled)
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        self.nc = int(nc)

    def __len__(self) -> int:
        return len(self.keys)

    def _img_path(self, key: str) -> str:
        return os.path.join(self.root, key)

    def _label_path(self, key: str) -> str:
        # default: replace extension with .txt and swap to labels_root while preserving subdirs
        stem, _ = os.path.splitext(key)
        return os.path.join(self.root, self.labels_root, stem + ".txt")

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        img_path = self._img_path(key)
        img = Image.open(img_path).convert("RGB")
        w0, h0 = img.size

        label_path = self._label_path(key)
        if os.path.exists(label_path):
            rows = []
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    parts = s.split()
                    if len(parts) != 5:
                        continue
                    rows.append([float(x) for x in parts])
            arr = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)
        else:
            arr = np.zeros((0, 5), dtype=np.float32)

        labels = arr[:, 0].astype(np.int64) if arr.shape[0] else np.zeros((0,), dtype=np.int64)
        xywhn = arr[:, 1:5] if arr.shape[0] else np.zeros((0, 4), dtype=np.float32)
        boxes = xywhn_to_xyxy_abs(xywhn, w=w0, h=h0) if arr.shape[0] else np.zeros((0, 4), dtype=np.float32)

        ratio = 1.0
        pad = (0, 0)
        if self.letterbox_enabled:
            img, ratio, pad = letterbox(img, self.input_size)
            boxes = letterbox_boxes_xyxy(boxes, ratio=ratio, pad=pad)
        else:
            img = img.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)
            sx = self.input_size[1] / w0
            sy = self.input_size[0] / h0
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        h, w = self.input_size
        boxes = clip_boxes_xyxy(boxes, w=w, h=h)
        m = valid_boxes_xyxy(boxes)
        boxes = boxes[m]
        labels = labels[m]

        # clamp labels to [0, nc-1] (defensive)
        if self.class_remap is not None and labels.size:
            mapped = []
            keep = []
            for i, c in enumerate(labels.tolist()):
                if int(c) in self.class_remap:
                    mapped.append(int(self.class_remap[int(c)]))
                    keep.append(i)
            if keep:
                keep = np.asarray(keep, dtype=np.int64)
                boxes = boxes[keep]
                labels = np.asarray(mapped, dtype=np.int64)
            else:
                boxes = boxes[:0]
                labels = labels[:0]
        labels = np.clip(labels, 0, self.nc - 1).astype(np.int64)

        arr_img = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arr_img = (arr_img - self.mean) / self.std
        image_t = torch.from_numpy(arr_img).float()

        target: Dict[str, Any] = {
            "boxes": torch.from_numpy(boxes).float(),
            "labels": torch.from_numpy(labels).long(),
            "image_id": idx,
            "path": key,
            "orig_size": (h0, w0),
            "input_size": self.input_size,
            "letterbox": {
                "ratio": float(ratio),
                "pad": (int(pad[0]), int(pad[1])),
            },
        }
        return image_t, target
