from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


def undersample_random(keys: List[str], *, keep_ratio: float, seed: int) -> List[str]:
    r = np.random.RandomState(seed)
    idx = np.arange(len(keys))
    r.shuffle(idx)
    k = int(round(len(keys) * float(keep_ratio)))
    return [keys[i] for i in idx[:k].tolist()]


def oversample_detection_duplicate(
    keys: List[str],
    dominant_classes: np.ndarray,
    *,
    seed: int,
) -> List[str]:
    """
    Detection-aware "oversampling" fallback when SMOTE isn't applicable.

    SMOTE is designed for vector features; for object detection (images + boxes),
    a practical baseline is to duplicate minority-class images.
    """
    r = np.random.RandomState(seed)
    cnt = Counter(int(c) for c in dominant_classes.tolist() if c >= 0)
    if not cnt:
        return keys
    max_n = max(cnt.values())
    out = list(keys)
    for cls, n in cnt.items():
        if n >= max_n:
            continue
        # duplicate random samples from this class
        idxs = np.where(dominant_classes == cls)[0]
        if idxs.size == 0:
            continue
        need = max_n - n
        picks = r.choice(idxs, size=int(need), replace=True)
        out.extend([keys[int(i)] for i in picks.tolist()])
    r.shuffle(out)
    return out

