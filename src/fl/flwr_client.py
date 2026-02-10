from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import flwr as fl
except Exception:  # pragma: no cover
    fl = None

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
from ..eval.evaluator import Evaluator
from ..models.factory import make_model
from ..utils.seed import seed_everything


class FlwrDetClient(fl.client.NumPyClient):  # type: ignore[misc]
    def __init__(self, *, cid: int, cfg: RootConfig) -> None:
        self.cid = int(cid)
        self.cfg = cfg

        # Make each client deterministic yet distinct.
        seed_everything(cfg.exp.seed + 1000 * (self.cid + 1))

        self.device = torch.device(cfg.exp.device)
        self.model = make_model(cfg.model, cfg.fl, cfg.preprocess, device_str=cfg.exp.device)

        keys = load_client_partition(cfg.data.client_partitions_dir, self.cid)
        base_ds = make_dataset(
            dataset_id=cfg.data.dataset_id,
            keys=keys,
            data_cfg=cfg.data,
            preprocess_cfg=cfg.preprocess,
            model_cfg=cfg.model,
            seed=cfg.exp.seed + 17 * (self.cid + 1),
        )

        # Apply poisoning only for attacker clients.
        if cfg.attack.enabled and (self.cid in set(cfg.attack.attacker_clients)):
            attack = make_attack(cfg.attack, cfg.eval, client_id=self.cid)
            if attack is None:
                ds = base_ds
            else:
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
                    client_id=self.cid,
                )
        else:
            ds = base_ds

        if cfg.augment.enabled:
            ds = AugmentedDataset(
                ds,
                augment=cfg.augment,
                preprocess=cfg.preprocess,
                seed=cfg.exp.seed + 4444 * (self.cid + 1),
            )

        self.train_loader = DataLoader(
            ds,
            batch_size=cfg.fl.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            collate_fn=detection_collate,
        )

    def get_parameters(self, config: Dict[str, Any]):
        return self.model.get_weights()

    def fit(self, parameters, config: Dict[str, Any]):
        self.model.set_weights(parameters)
        local_epochs = int(self.cfg.fl.local_epochs)
        metrics: Dict[str, float] = {}
        for e in range(local_epochs):
            m = self.model.train_one_epoch(self.train_loader)
            for k, v in m.items():
                metrics[f"train/{k}"] = float(v)
        return self.model.get_weights(), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config: Dict[str, Any]):
        # Client-side eval is not used in this baseline (centralized eval on server).
        self.model.set_weights(parameters)
        return 0.0, 0, {}


def make_client_fn(cfg: RootConfig):
    if fl is None:
        raise RuntimeError("flwr is not installed. Install requirements and retry.")

    def client_fn(cid: str):
        return FlwrDetClient(cid=int(cid), cfg=cfg)

    return client_fn
