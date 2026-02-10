from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..letterbox import letterbox, letterbox_boxes_xyxy
from ..targets import clip_boxes_xyxy, valid_boxes_xyxy
from .base import DetectionDataset


class SyntheticDetDataset(DetectionDataset):
    """
    Deterministic synthetic detection dataset for smoke tests.

    - Generates 1..3 boxes per image
    - Classes: 0..(nc-1)
    - Boxes are drawn into the image to create a learnable signal
    """

    def __init__(
        self,
        keys: List[str],
        *,
        seed: int,
        input_size: Tuple[int, int],
        letterbox_enabled: bool,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        nc: int,
    ) -> None:
        self.keys = keys
        self.seed = int(seed)
        self.input_size = input_size
        self.letterbox_enabled = bool(letterbox_enabled)
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        self.nc = int(nc)

    def __len__(self) -> int:
        return len(self.keys)

    def _rng(self, idx: int) -> np.random.RandomState:
        # Stable across runs & platforms.
        return np.random.RandomState(self.seed + idx * 10007)

    def __getitem__(self, idx: int):
        r = self._rng(idx)
        h0, w0 = 360, 640
        img = Image.fromarray(r.randint(0, 35, size=(h0, w0, 3), dtype=np.uint8), mode="RGB")

        n = int(r.randint(1, 4))
        boxes = []
        labels = []
        draw = ImageDraw.Draw(img)
        for _ in range(n):
            cls = int(r.randint(0, self.nc))
            x1 = int(r.randint(0, w0 - 80))
            y1 = int(r.randint(0, h0 - 80))
            x2 = int(r.randint(x1 + 20, min(w0, x1 + 160)))
            y2 = int(r.randint(y1 + 20, min(h0, y1 + 160)))
            boxes.append([x1, y1, x2, y2])
            labels.append(cls)
            # draw filled rectangle with class-coded color
            color = [(255, 50, 50), (255, 255, 50), (50, 255, 50)][cls % 3]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.rectangle([x1 + 3, y1 + 3, x2 - 3, y2 - 3], fill=tuple(int(c * 0.25) for c in color))

        boxes = np.asarray(boxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

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

        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arr = (arr - self.mean) / self.std
        image_t = torch.from_numpy(arr).float()

        target: Dict[str, Any] = {
            "boxes": torch.from_numpy(boxes).float(),
            "labels": torch.from_numpy(labels).long(),
            "image_id": idx,
            "path": self.keys[idx],
            "orig_size": (h0, w0),
            "input_size": self.input_size,
            "letterbox": {
                "ratio": float(ratio),
                "pad": (int(pad[0]), int(pad[1])),
            },
        }
        return image_t, target

