from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch


@dataclass(frozen=True)
class TriggerSpec:
    pattern: str
    size_px: int
    alpha: float
    position: str
    apply_prob: float = 1.0


class Attack:
    name: str

    def poison_batch(
        self,
        images: List[torch.Tensor],
        targets: List[Dict[str, Any]],
        trigger: TriggerSpec,
        *,
        rng: torch.Generator,
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:  # pragma: no cover
        raise NotImplementedError

