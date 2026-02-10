from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .splits import read_split_file


def _write_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")


def random_split(
    keys: List[str],
    *,
    seed: int,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Dict[str, List[str]]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")
    r = np.random.RandomState(seed)
    idx = np.arange(len(keys))
    r.shuffle(idx)
    n = len(keys)
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    tr = [keys[i] for i in idx[:n_tr]]
    va = [keys[i] for i in idx[n_tr : n_tr + n_va]]
    te = [keys[i] for i in idx[n_tr + n_va :]]
    return {"train": tr, "val": va, "test": te}


def _read_yolo_labels_for_key(root: str, labels_root: str, key: str) -> np.ndarray:
    stem, _ = os.path.splitext(key)
    p = os.path.join(root, labels_root, stem + ".txt")
    if not os.path.exists(p):
        return np.zeros((0,), dtype=np.int64)
    labels = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 5:
                continue
            labels.append(int(float(parts[0])))
    return np.asarray(labels, dtype=np.int64)


def dominant_class_per_image(
    keys: List[str],
    *,
    root: str,
    labels_root: str,
    nc: int,
) -> np.ndarray:
    """
    Returns per-image dominant class id (0..nc-1), or -1 if no labels.
    """
    dom = np.full((len(keys),), -1, dtype=np.int64)
    for i, k in enumerate(keys):
        y = _read_yolo_labels_for_key(root, labels_root, k)
        if y.size == 0:
            continue
        y = np.clip(y, 0, nc - 1)
        vals, cnts = np.unique(y, return_counts=True)
        dom[i] = int(vals[int(np.argmax(cnts))])
    return dom


def dirichlet_partition(
    keys: List[str],
    dom_cls: np.ndarray,
    *,
    num_clients: int,
    nc: int,
    alpha: float,
    seed: int,
) -> Dict[int, List[str]]:
    """
    Non-IID partition via Dirichlet over classes using per-image dominant class.
    """
    r = np.random.RandomState(seed)
    cls_to_indices: Dict[int, List[int]] = {c: [] for c in range(nc)}
    for i, c in enumerate(dom_cls.tolist()):
        if 0 <= c < nc:
            cls_to_indices[c].append(i)

    for c in range(nc):
        r.shuffle(cls_to_indices[c])

    client_keys: Dict[int, List[str]] = {cid: [] for cid in range(num_clients)}
    for c in range(nc):
        idxs = cls_to_indices[c]
        if not idxs:
            continue
        proportions = r.dirichlet([alpha] * num_clients)
        # Convert proportions to counts and fix rounding.
        counts = (proportions * len(idxs)).astype(int)
        while counts.sum() < len(idxs):
            counts[int(r.randint(0, num_clients))] += 1
        while counts.sum() > len(idxs):
            j = int(r.randint(0, num_clients))
            if counts[j] > 0:
                counts[j] -= 1
        start = 0
        for cid in range(num_clients):
            take = int(counts[cid])
            if take <= 0:
                continue
            take_idxs = idxs[start : start + take]
            start += take
            client_keys[cid].extend([keys[i] for i in take_idxs])

    # Shuffle each client list for stable training order.
    for cid in range(num_clients):
        r.shuffle(client_keys[cid])
    return client_keys


def write_splits_and_partitions(
    *,
    all_keys: List[str],
    out_splits_dir: str,
    out_partitions_dir: str,
    seed: int,
    ratios: Tuple[float, float, float],
    num_clients: int,
    alpha: float,
    root_for_labels: Optional[str],
    labels_root: str,
    nc: int,
) -> None:
    splits = random_split(all_keys, seed=seed, ratios=ratios)
    _write_lines(os.path.join(out_splits_dir, "train.txt"), splits["train"])
    _write_lines(os.path.join(out_splits_dir, "val.txt"), splits["val"])
    _write_lines(os.path.join(out_splits_dir, "test.txt"), splits["test"])

    # Partition only train split by default.
    train_keys = splits["train"]
    if root_for_labels is None:
        # If labels aren't available, fall back to uniform random partition.
        r = np.random.RandomState(seed)
        r.shuffle(train_keys)
        client_keys = {cid: [] for cid in range(num_clients)}
        for i, k in enumerate(train_keys):
            client_keys[i % num_clients].append(k)
    else:
        dom = dominant_class_per_image(train_keys, root=root_for_labels, labels_root=labels_root, nc=nc)
        client_keys = dirichlet_partition(
            train_keys, dom, num_clients=num_clients, nc=nc, alpha=alpha, seed=seed
        )

    os.makedirs(out_partitions_dir, exist_ok=True)
    for cid in range(num_clients):
        _write_lines(os.path.join(out_partitions_dir, f"client_{cid}.txt"), client_keys[cid])


def make_splits_cli() -> None:
    """
    Minimal CLI (optional):
      python -m src.data.split_generate --all splits/all.txt --out_splits splits --out_parts splits/clients_alpha0.3_seed42 --seed 42 --num_clients 50 --alpha 0.3
    """
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--all", required=True, help="Path to master list file (one image path per line).")
    p.add_argument("--out_splits", required=True)
    p.add_argument("--out_parts", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num_clients", type=int, required=True)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--ratios", type=str, default="0.7,0.15,0.15")
    p.add_argument("--root_for_labels", type=str, default=None)
    p.add_argument("--labels_root", type=str, default="labels")
    p.add_argument("--nc", type=int, required=True)
    args = p.parse_args()

    ratios = tuple(float(x) for x in args.ratios.split(","))
    all_keys = read_split_file(args.all)
    write_splits_and_partitions(
        all_keys=all_keys,
        out_splits_dir=args.out_splits,
        out_partitions_dir=args.out_parts,
        seed=args.seed,
        ratios=ratios,  # 7:1.5:1.5
        num_clients=args.num_clients,
        alpha=args.alpha,
        root_for_labels=args.root_for_labels,
        labels_root=args.labels_root,
        nc=args.nc,
    )


if __name__ == "__main__":
    make_splits_cli()

