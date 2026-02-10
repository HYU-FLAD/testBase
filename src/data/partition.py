from __future__ import annotations

import os
from typing import Dict, List

from ..utils.hashing import sha256_lines


def load_client_partition(partitions_dir: str, cid: int) -> List[str]:
    path = os.path.join(partitions_dir, f"client_{cid}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    return lines


def hash_client_partition(sample_keys: List[str]) -> str:
    return sha256_lines(sample_keys)


def load_all_partition_hashes(partitions_dir: str, num_clients: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for cid in range(num_clients):
        keys = load_client_partition(partitions_dir, cid)
        out[str(cid)] = hash_client_partition(keys)
    return out

