from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict
from typing import Any, Dict, Optional

from .hashing import sha256_file, sha256_text


def resolve_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"


def get_torch_env() -> Dict[str, Any]:
    try:
        import torch

        return {
            "torch": getattr(torch, "__version__", "UNKNOWN"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda": getattr(torch.version, "cuda", None),
            "cudnn": getattr(torch.backends.cudnn, "version", lambda: None)(),
        }
    except Exception:
        return {"torch": None, "cuda_available": False, "cuda": None, "cudnn": None}


def make_meta(
    *,
    exp_id: str,
    seed: int,
    git_commit: str,
    config_resolved: Dict[str, Any],
    split_hashes: Dict[str, str],
    partition_hashes: Dict[str, str],
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "exp_id": exp_id,
        "seed": seed,
        "git_commit": git_commit,
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "torch_env": get_torch_env(),
        "config_sha256": sha256_text(json.dumps(config_resolved, sort_keys=True)),
        "split_hashes": split_hashes,
        "client_partition_hashes": partition_hashes,
    }
    return meta


def dump_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def dump_yaml(path: str, obj: Dict[str, Any]) -> None:
    import yaml

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

