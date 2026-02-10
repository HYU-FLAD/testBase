from __future__ import annotations

import os
from typing import List

import numpy as np

from ..data.split_generate import write_splits_and_partitions


def ensure_synth_splits(
    *,
    split_train: str,
    split_val: str,
    split_test: str,
    partitions_dir: str,
    seed: int,
    num_clients: int,
    nc: int,
    n_samples: int = 200,
    alpha: float = 0.3,
) -> None:
    """
    Creates split files + client partitions if missing.
    Keys are synthetic identifiers (no filesystem images required).
    """
    if os.path.exists(split_train) and os.path.exists(split_val) and os.path.exists(split_test):
        # Still ensure partitions exist.
        ok = True
        for cid in range(num_clients):
            if not os.path.exists(os.path.join(partitions_dir, f"client_{cid}.txt")):
                ok = False
                break
        if ok:
            return

    all_keys: List[str] = [f"synthetic_{i:06d}" for i in range(int(n_samples))]
    out_splits_dir = os.path.dirname(os.path.abspath(split_train))
    out_parts_dir = os.path.abspath(partitions_dir)
    write_splits_and_partitions(
        all_keys=all_keys,
        out_splits_dir=out_splits_dir,
        out_partitions_dir=out_parts_dir,
        seed=int(seed),
        ratios=(0.7, 0.15, 0.15),
        num_clients=int(num_clients),
        alpha=float(alpha),
        root_for_labels=None,
        labels_root="labels",
        nc=int(nc),
    )

