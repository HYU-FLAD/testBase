from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from .base import Attack, TriggerSpec
from .trigger_attacks import TriggerGenerationAttack


class DBASkeletonAttack(Attack):
    """
    DBA (Distributed Backdoor Attack) skeleton.

    TODO: replace with your DBA implementation (multiple trigger parts across clients, aggregation logic, etc.).
    MVP behavior: falls back to trigger_generation with per-client trigger pattern variant.
    """

    name = "dba"

    def __init__(self, *, target_class: int, client_id: int) -> None:
        self.target_class = int(target_class)
        self.client_id = int(client_id)
        self._base = TriggerGenerationAttack(target_class=target_class)

    def poison_batch(self, images, targets, trigger: TriggerSpec, *, rng: torch.Generator):
        # Simple per-client pattern switch as a placeholder for "distributed" trigger.
        if (self.client_id % 2) == 0:
            pattern = "checker"
        else:
            pattern = "solid_red"
        trig2 = TriggerSpec(
            pattern=pattern,
            size_px=trigger.size_px,
            alpha=trigger.alpha,
            position=trigger.position,
            apply_prob=trigger.apply_prob,
        )
        return self._base.poison_batch(images, targets, trig2, rng=rng)

