from __future__ import annotations

import os
from typing import List


def read_split_file(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out

