from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import flwr as fl
except Exception:  # pragma: no cover
    fl = None

import torch
from torch.utils.data import DataLoader

from ..config.schema import RootConfig
from ..data.datasets.base import detection_collate
from ..data.factory import make_dataset
from ..data.splits import read_split_file
from ..eval.evaluator import Evaluator
from ..models.factory import make_model
from ..utils.logger import CheckpointTracker, jsonl_append
from ..utils.seed import seed_everything


def make_strategy_and_eval_fn(cfg: RootConfig, *, out_dir: str):
    if fl is None:
        raise RuntimeError("flwr is not installed. Install requirements and retry.")

    seed_everything(cfg.exp.seed)
    device = torch.device(cfg.exp.device)

    # Centralized validation loader
    val_keys = read_split_file(cfg.data.split_files.val)
    val_ds = make_dataset(
        dataset_id=cfg.data.dataset_id,
        keys=val_keys,
        data_cfg=cfg.data,
        preprocess_cfg=cfg.preprocess,
        model_cfg=cfg.model,
        seed=cfg.exp.seed + 123,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.fl.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=detection_collate,
    )

    evaluator = Evaluator(
        eval_cfg=cfg.eval,
        preprocess_cfg=cfg.preprocess,
        model_cfg=cfg.model,
        attack_cfg=cfg.attack,
        trigger_cfg=cfg.trigger,
        device=device,
    )

    global_model = make_model(cfg.model, cfg.fl, cfg.preprocess, device_str=cfg.exp.device)
    ckpt = CheckpointTracker(out_dir=out_dir, best_metric_key="map50_95")

    def evaluate_fn(server_round: int, parameters, config):
        global_model.set_weights(parameters)
        metrics = evaluator.evaluate(global_model, val_loader)
        row = {"round": int(server_round), **{k: float(v) for k, v in metrics.items()}}
        jsonl_append(path=f"{out_dir}/round.jsonl", row=row)
        jsonl_append(path=f"{out_dir}/round_{int(server_round):03d}.jsonl", row=row)
        # For MVP, treat each FL round as an "epoch" for logging parity.
        jsonl_append(path=f"{out_dir}/epoch_{int(server_round):03d}.jsonl", row=row)

        # Save best/last checkpoints based on map50_95
        ckpt.save_last(global_model)
        ckpt.maybe_update_best(float(metrics.get("map50_95", 0.0)), global_model)
        loss = 1.0 - float(metrics.get("map50_95", 0.0))
        return loss, {k: float(v) for k, v in metrics.items()}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=float(cfg.fl.clients_per_round) / float(cfg.data.num_clients),
        fraction_evaluate=0.0,
        min_fit_clients=int(cfg.fl.clients_per_round),
        min_available_clients=int(cfg.data.num_clients),
        on_evaluate_config_fn=lambda r: {},
        evaluate_fn=evaluate_fn,
    )
    return strategy


def run_flwr_fedavg(cfg: RootConfig, *, out_dir: str) -> None:
    if fl is None:
        raise RuntimeError("flwr is not installed. Install requirements and retry.")

    from .flwr_client import make_client_fn

    client_fn = make_client_fn(cfg)
    strategy = make_strategy_and_eval_fn(cfg, out_dir=out_dir)

    # Simulation mode (requires flwr simulation deps; usually Ray under the hood).
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=int(cfg.data.num_clients),
        config=fl.server.ServerConfig(num_rounds=int(cfg.fl.rounds)),
        strategy=strategy,
        client_resources=None,  # set e.g. {"num_cpus":1,"num_gpus":0} if needed
    )
