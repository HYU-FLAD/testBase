from __future__ import annotations

from typing import Dict, Optional


def get_class_remap(dataset_id: str) -> Optional[Dict[int, int]]:
    """
    Required unified class ids:
    - LISA: red(0), yellow(1), green(2)
    - BDD100K: red(0), yellow(1), green(2), sign(3), person(4)

    MVP assumption: your dataset export already uses these ids, so remap is identity.
    TODO: implement true remap from original class taxonomy if your raw labels differ.
    """
    ds = dataset_id.lower()
    if ds == "lisa":
        return {0: 0, 1: 1, 2: 2}
    if ds == "bdd100k":
        return {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    return None

