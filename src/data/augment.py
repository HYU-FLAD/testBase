from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config.schema import AugmentConfig, PreprocessConfig


class AugmentedDataset(Dataset):
    """
    Common augmentation wrapper (framework-agnostic).

    MVP implementation:
    - color_jitter: lightweight brightness/contrast jitter (in tensor space)
    - random_affine/mosaic/mixup: TODO (kept as no-ops for now)

    Important: augmentation is applied only in the shared data pipeline for fairness.
    """

    def __init__(
        self,
        base: Dataset,
        *,
        augment: AugmentConfig,
        preprocess: PreprocessConfig,
        seed: int,
    ) -> None:
        self.base = base
        self.augment = augment
        self.preprocess = preprocess
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base)

    def _rng(self, idx: int) -> np.random.RandomState:
        return np.random.RandomState(self.seed + 20011 * (idx + 1))

    def __getitem__(self, idx: int):
        img, tgt = self.base[idx]
        if not self.augment.enabled:
            return img, tgt

        r = self._rng(idx)

        # color jitter (brightness/contrast) in normalized tensor space
        if r.rand() < float(self.augment.color_jitter_p):
            # apply small global affine transform per-channel
            b = float(r.uniform(-0.05, 0.05))
            c = float(r.uniform(0.9, 1.1))
            img = (img * c + b).to(img.dtype)

        # TODO: mosaic/mixup/random_affine (needs careful box transforms in letterboxed space)
        return img, tgt

