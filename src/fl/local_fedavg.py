from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..attacks.base import TriggerSpec
from ..attacks.poisoned import PoisonedDataset
from ..attacks.registry import make_attack
from ..config.schema import RootConfig
from ..data.datasets.base import detection_collate
from ..data.augment import AugmentedDataset
from ..data.factory import make_dataset
from ..data.partition import load_client_partition
from ..data.splits import read_split_file
from ..eval.evaluator import Evaluator
from ..models.factory import make_model
from ..utils.logger import CheckpointTracker, jsonl_append
from ..utils.seed import seed_everything


def _weighted_average(weights_list: List[List[np.ndarray]], n_list: List[int]) -> List[np.ndarray]:
    tot = float(sum(n_list))
    if tot <= 0:
        raise ValueError("No examples to aggregate")
    agg: List[np.ndarray] = []
    for layer_i in range(len(weights_list[0])):
        s = None
        for w, n in zip(weights_list, n_list):
            x = w[layer_i].astype(np.float32) * (float(n) / tot)
            s = x if s is None else (s + x)
        agg.append(s.astype(weights_list[0][layer_i].dtype, copy=False))
    return agg


def run_local_fedavg(cfg: RootConfig, *, out_dir: str) -> None:
    """
    Deterministic FedAvg fallback when `flwr` is not installed.
    This keeps the same ModelAdapter contract + shared preprocessing/eval.
    """
    seed_everything(cfg.exp.seed)
    r = np.random.RandomState(cfg.exp.seed + 777)

    device = torch.device(cfg.exp.device)
    global_model = make_model(cfg.model, cfg.fl, cfg.preprocess, device_str=cfg.exp.device)

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
    ckpt = CheckpointTracker(out_dir=out_dir, best_metric_key="map50_95")

    all_clients = list(range(cfg.data.num_clients))
    for rnd in range(1, int(cfg.fl.rounds) + 1):
        chosen = r.choice(all_clients, size=int(cfg.fl.clients_per_round), replace=False).tolist()
        weights_before = global_model.get_weights()

        client_weights = []
        client_ns = []

        for cid in chosen:
            # per-client deterministic init
            seed_everything(cfg.exp.seed + 1000 * (cid + 1) + 10 * rnd)
            model = make_model(cfg.model, cfg.fl, cfg.preprocess, device_str=cfg.exp.device)
            model.set_weights(weights_before)

            keys = load_client_partition(cfg.data.client_partitions_dir, int(cid))
            base_ds = make_dataset(
                dataset_id=cfg.data.dataset_id,
                keys=keys,
                data_cfg=cfg.data,
                preprocess_cfg=cfg.preprocess,
                model_cfg=cfg.model,
                seed=cfg.exp.seed + 17 * (int(cid) + 1),
            )
            if cfg.attack.enabled and (int(cid) in set(cfg.attack.attacker_clients)):
                attack = make_attack(cfg.attack, cfg.eval, client_id=int(cid))
                if attack is not None:
                    trig = TriggerSpec(
                        pattern=cfg.trigger.pattern,
                        size_px=cfg.trigger.size_px,
                        alpha=cfg.trigger.alpha,
                        position=cfg.trigger.position,
                        apply_prob=cfg.trigger.apply_prob,
                    )
                    ds = PoisonedDataset(
                        base_ds,
                        attack=attack,
                        trigger=trig,
                        poison_rate=cfg.attack.poison_rate,
                        seed=cfg.exp.seed,
                        client_id=int(cid),
                    )
                else:
                    ds = base_ds
            else:
                ds = base_ds

            if cfg.augment.enabled:
                ds = AugmentedDataset(
                    ds,
                    augment=cfg.augment,
                    preprocess=cfg.preprocess,
                    seed=cfg.exp.seed + 4444 * (int(cid) + 1) + 10 * rnd,
                )

            loader = DataLoader(
                ds,
                batch_size=cfg.fl.batch_size,
                shuffle=True,
                num_workers=cfg.data.num_workers,
                collate_fn=detection_collate,
            )

            for _ in range(int(cfg.fl.local_epochs)):
                model.train_one_epoch(loader)

            client_weights.append(model.get_weights())
            client_ns.append(len(ds))

        new_weights = _weighted_average(client_weights, client_ns)
        global_model.set_weights(new_weights)

        metrics = evaluator.evaluate(global_model, val_loader)
        row = {"round": int(rnd), **{k: float(v) for k, v in metrics.items()}}
        jsonl_append(path=f"{out_dir}/round.jsonl", row=row)
        jsonl_append(path=f"{out_dir}/round_{int(rnd):03d}.jsonl", row=row)
        jsonl_append(path=f"{out_dir}/epoch_{int(rnd):03d}.jsonl", row=row)
        ckpt.save_last(global_model)
        ckpt.maybe_update_best(float(metrics.get("map50_95", 0.0)), global_model)
