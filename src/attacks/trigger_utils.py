from __future__ import annotations

from typing import Tuple

import torch


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
        # center
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(w - 1, x1 + size)
        y2 = min(h - 1, y1 + size)
    return int(x1), int(y1), int(x2), int(y2)


def apply_trigger(
    image: torch.Tensor,
    *,
    trigger_pattern: str,
    size_px: int,
    alpha: float,
    position: str,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    image: float tensor [C,H,W] (normalized or not; we just overlay).
    Returns: poisoned_image, trigger_box_xyxy (int coords in image space).
    """
    assert image.ndim == 3 and image.shape[0] == 3
    c, h, w = image.shape
    x1, y1, x2, y2 = _trigger_xyxy(h, w, size_px, position)

    out = image.clone()
    patch_h = max(1, y2 - y1)
    patch_w = max(1, x2 - x1)
    if trigger_pattern == "checker":
        yy = torch.arange(patch_h, device=image.device)[:, None]
        xx = torch.arange(patch_w, device=image.device)[None, :]
        mask = ((yy // 4 + xx // 4) % 2).float()  # coarse checker
        patch = torch.stack([mask, 1.0 - mask, mask * 0.0], dim=0)  # RGB-ish
    elif trigger_pattern == "solid_red":
        patch = torch.zeros((3, patch_h, patch_w), device=image.device)
        patch[0, :, :] = 1.0
    else:
        # fallback
        patch = torch.ones((3, patch_h, patch_w), device=image.device) * 0.5

    a = float(alpha)
    out[:, y1:y2, x1:x2] = (1.0 - a) * out[:, y1:y2, x1:x2] + a * patch
    return out, (x1, y1, x2, y2)


def roi_box_from_trigger(
    trigger_xyxy: Tuple[int, int, int, int],
    *,
    h: int,
    w: int,
    expand_px: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = trigger_xyxy
    x1 = max(0, x1 - int(expand_px))
    y1 = max(0, y1 - int(expand_px))
    x2 = min(w - 1, x2 + int(expand_px))
    y2 = min(h - 1, y2 + int(expand_px))
    return int(x1), int(y1), int(x2), int(y2)

