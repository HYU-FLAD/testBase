from __future__ import annotations

import copy
import os
from typing import Any, Dict, Tuple

import yaml

from .schema import RootConfig, parse_config
from ..utils.meta import resolve_git_commit


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str) -> Tuple[RootConfig, Dict[str, Any]]:
    raw = load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a dict")
    _ = parse_config(raw)  # validate schema early

    resolved = copy.deepcopy(raw)
    if isinstance(resolved.get("exp", {}), dict) and resolved["exp"].get("git_commit", "AUTO") == "AUTO":
        resolved["exp"]["git_commit"] = resolve_git_commit()
    resolved["_config_path"] = os.path.abspath(path)

    cfg = parse_config(resolved)
    return cfg, resolved
