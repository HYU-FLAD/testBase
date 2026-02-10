from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def letterbox(
    img: Image.Image,
    new_shape: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, float, Tuple[int, int]]:
    """
    Resize with unchanged aspect ratio using padding.
    Returns: resized_img, ratio, (dw, dh) padding offsets.
    """
    w0, h0 = img.size
    new_h, new_w = new_shape
    r = min(new_w / w0, new_h / h0)
    w1, h1 = int(round(w0 * r)), int(round(h0 * r))
    img_resized = img.resize((w1, h1), Image.BILINEAR)

    canvas = Image.new("RGB", (new_w, new_h), color)
    dw = (new_w - w1) // 2
    dh = (new_h - h1) // 2
    canvas.paste(img_resized, (dw, dh))
    return canvas, r, (dw, dh)


def letterbox_boxes_xyxy(
    boxes: np.ndarray,
    *,
    ratio: float,
    pad: Tuple[int, int],
) -> np.ndarray:
    dw, dh = pad
    b = boxes.copy()
    b[:, [0, 2]] = b[:, [0, 2]] * ratio + dw
    b[:, [1, 3]] = b[:, [1, 3]] * ratio + dh
    return b

