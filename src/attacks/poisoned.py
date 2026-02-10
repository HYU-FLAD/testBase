from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import Attack, TriggerSpec


class PoisonedDataset(Dataset):
    """
    Deterministic poisoning wrapper.

    poison_rate is defined as: fraction of the *attacker client's local samples* that are poisoned.
    Poison selection is stable given (seed, client_id, dataset length).
    """

    def __init__(
        self,
        base: Dataset,
        *,
        attack: Attack,
        trigger: TriggerSpec,
        poison_rate: float,
        seed: int,
        client_id: int,
    ) -> None:
        self.base = base
        self.attack = attack
        self.trigger = trigger
        self.poison_rate = float(poison_rate)
        self.seed = int(seed)
        self.client_id = int(client_id)

        n = len(base)
        k = int(round(n * self.poison_rate))
        r = np.random.RandomState(self.seed + 1337 * (self.client_id + 1))
        idxs = np.arange(n)
        r.shuffle(idxs)
        self.poison_indices = set(int(i) for i in idxs[:k].tolist())

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, tgt = self.base[idx]
        if int(idx) not in self.poison_indices:
            return img, tgt

        # Attack interface operates on batches; wrap single sample.
        g = torch.Generator(device=img.device)
        g.manual_seed(self.seed + 99991 * (self.client_id + 1) + int(idx))
        imgs, tgts = self.attack.poison_batch([img], [tgt], self.trigger, rng=g)
        return imgs[0], tgts[0]

