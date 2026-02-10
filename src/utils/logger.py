from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def now_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def jsonl_append(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


@dataclass
class CheckpointTracker:
    out_dir: str
    best_metric_key: str = "map50_95"
    best_metric: float = float("-inf")

    def maybe_update_best(self, metric: float, model_adapter: Any) -> bool:
        if metric > self.best_metric:
            self.best_metric = metric
            model_adapter.save(os.path.join(self.out_dir, "best.pt"))
            return True
        return False

    def save_last(self, model_adapter: Any) -> None:
        model_adapter.save(os.path.join(self.out_dir, "last.pt"))

