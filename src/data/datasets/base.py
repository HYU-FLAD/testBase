from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    """Common dataset output contract used across frameworks."""

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover
        raise NotImplementedError


def detection_collate(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]):
    images, targets = zip(*batch)
    return list(images), list(targets)

