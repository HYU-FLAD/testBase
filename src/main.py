from __future__ import annotations

import argparse
import os

from .config.loader import load_config
from .data.partition import load_all_partition_hashes
from .utils.bootstrap import ensure_synth_splits
from .utils.hashing import sha256_file
from .utils.logger import ensure_dir, now_timestamp
from .utils.meta import dump_json, dump_yaml, make_meta
from .utils.seed import seed_everything


def _split_hashes(cfg) -> dict:
    out = {}
    for k, p in [
        ("train", cfg.data.split_files.train),
        ("val", cfg.data.split_files.val),
        ("test", cfg.data.split_files.test),
    ]:
        if os.path.exists(p):
            out[k] = sha256_file(p)
        else:
            out[k] = "MISSING"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to experiment YAML")
    args = ap.parse_args()

    cfg, cfg_resolved = load_config(args.config)
    seed_everything(cfg.exp.seed)

    # Bootstrap splits/partitions for synthetic dataset.
    if cfg.data.dataset_id.lower() == "synthetic":
        ensure_synth_splits(
            split_train=cfg.data.split_files.train,
            split_val=cfg.data.split_files.val,
            split_test=cfg.data.split_files.test,
            partitions_dir=cfg.data.client_partitions_dir,
            seed=cfg.exp.seed,
            num_clients=cfg.data.num_clients,
            nc=cfg.model.nc,
            n_samples=200,
            alpha=0.3,
        )

    ts = now_timestamp()
    exp_id = f"{cfg.model.type}_{cfg.model.variant}_{ts}"
    out_dir = os.path.abspath(os.path.join("logs", exp_id))
    ensure_dir(out_dir)

    # Dump resolved config + meta for reproducibility.
    dump_yaml(os.path.join(out_dir, "config_resolved.yaml"), cfg_resolved)
    split_hashes = _split_hashes(cfg)
    part_hashes = load_all_partition_hashes(cfg.data.client_partitions_dir, cfg.data.num_clients)
    meta = make_meta(
        exp_id=exp_id,
        seed=cfg.exp.seed,
        git_commit=cfg.exp.git_commit,
        config_resolved=cfg_resolved,
        split_hashes=split_hashes,
        partition_hashes=part_hashes,
    )
    dump_json(os.path.join(out_dir, "meta.json"), meta)

    # Run FL
    if cfg.fl.algorithm.lower() != "fedavg":
        raise ValueError(f"Unsupported FL algorithm (MVP): {cfg.fl.algorithm}")

    flwr_ok = True
    try:
        import flwr  # noqa: F401
    except Exception:
        flwr_ok = False

    if flwr_ok:
        from .fl.flwr_server import run_flwr_fedavg

        run_flwr_fedavg(cfg, out_dir=out_dir)
    else:
        # Fallback for environments without flwr installed (keeps the same contract).
        from .fl.local_fedavg import run_local_fedavg

        run_local_fedavg(cfg, out_dir=out_dir)


if __name__ == "__main__":
    main()
