from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _require(d: Dict[str, Any], key: str, path: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required key: {path}{key}")
    return d[key]


def _as_int(x: Any, path: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise ValueError(f"Expected int at {path}, got {type(x).__name__}")
    return x


def _as_float(x: Any, path: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"Expected float at {path}, got {type(x).__name__}")
    return float(x)


def _as_str(x: Any, path: str) -> str:
    if not isinstance(x, str):
        raise ValueError(f"Expected str at {path}, got {type(x).__name__}")
    return x


def _as_bool(x: Any, path: str) -> bool:
    if not isinstance(x, bool):
        raise ValueError(f"Expected bool at {path}, got {type(x).__name__}")
    return x


def _as_list(x: Any, path: str) -> list:
    if isinstance(x, tuple):
        return list(x)
    if not isinstance(x, list):
        raise ValueError(f"Expected list at {path}, got {type(x).__name__}")
    return x


@dataclass(frozen=True)
class ExpConfig:
    name: str
    seed: int
    git_commit: str = "AUTO"
    device: str = "cuda"
    precision: str = "fp16"  # fp16 / bf16 / fp32


@dataclass(frozen=True)
class SplitFiles:
    train: str
    val: str
    test: str


@dataclass(frozen=True)
class DataConfig:
    dataset_id: str
    split_files: SplitFiles
    client_partitions_dir: str
    num_clients: int
    root: str = "datasets"
    num_workers: int = 2
    labels_root: str = "labels"


@dataclass(frozen=True)
class PreprocessConfig:
    input_size: Tuple[int, int]
    normalize_mean: Tuple[float, float, float]
    normalize_std: Tuple[float, float, float]
    letterbox: bool = True


@dataclass(frozen=True)
class AugmentConfig:
    enabled: bool
    mosaic_p: float = 0.0
    mixup_p: float = 0.0
    random_affine_p: float = 0.0
    degrees: float = 0.0
    translate: float = 0.0
    scale: float = 0.0
    color_jitter_p: float = 0.0


@dataclass(frozen=True)
class OptimConfig:
    name: str
    lr: float
    momentum: float = 0.0
    wd: float = 0.0


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    warmup_epochs: int = 0


@dataclass(frozen=True)
class FLConfig:
    algorithm: str
    rounds: int
    clients_per_round: int
    local_epochs: int
    batch_size: int
    optimizer: OptimConfig
    scheduler: SchedulerConfig
    clip_norm: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    type: str  # yolo / rcnn / ...
    variant: str
    pretrained: bool
    nc: int


@dataclass(frozen=True)
class AttackConfig:
    enabled: bool
    attacker_clients: List[int] = field(default_factory=list)
    poison_rate: float = 0.0
    target_class: int = 0
    method: str = "trigger_generation"


@dataclass(frozen=True)
class TriggerConfig:
    pattern: str
    size_px: int
    alpha: float
    position: str
    apply_prob: float = 1.0


@dataclass(frozen=True)
class EvalConfig:
    conf_thresh: float
    nms_iou: float
    max_det: int
    iou_match: float
    asr_defs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class RootConfig:
    exp: ExpConfig
    data: DataConfig
    preprocess: PreprocessConfig
    augment: AugmentConfig
    fl: FLConfig
    model: ModelConfig
    attack: AttackConfig
    trigger: TriggerConfig
    eval: EvalConfig


def parse_config(d: Dict[str, Any]) -> RootConfig:
    exp_d = _require(d, "exp", ""); exp_d = exp_d if isinstance(exp_d, dict) else None
    if exp_d is None:
        raise ValueError("exp must be a dict")
    exp = ExpConfig(
        name=_as_str(_require(exp_d, "name", "exp."), "exp.name"),
        seed=_as_int(_require(exp_d, "seed", "exp."), "exp.seed"),
        git_commit=_as_str(exp_d.get("git_commit", "AUTO"), "exp.git_commit"),
        device=_as_str(exp_d.get("device", "cuda"), "exp.device"),
        precision=_as_str(exp_d.get("precision", "fp16"), "exp.precision"),
    )

    data_d = _require(d, "data", ""); data_d = data_d if isinstance(data_d, dict) else None
    if data_d is None:
        raise ValueError("data must be a dict")
    split_d = _require(data_d, "split_files", "data."); split_d = split_d if isinstance(split_d, dict) else None
    if split_d is None:
        raise ValueError("data.split_files must be a dict")
    split_files = SplitFiles(
        train=_as_str(_require(split_d, "train", "data.split_files."), "data.split_files.train"),
        val=_as_str(_require(split_d, "val", "data.split_files."), "data.split_files.val"),
        test=_as_str(_require(split_d, "test", "data.split_files."), "data.split_files.test"),
    )
    data = DataConfig(
        dataset_id=_as_str(_require(data_d, "dataset_id", "data."), "data.dataset_id"),
        root=_as_str(data_d.get("root", "datasets"), "data.root"),
        split_files=split_files,
        client_partitions_dir=_as_str(_require(data_d, "client_partitions_dir", "data."), "data.client_partitions_dir"),
        num_clients=_as_int(_require(data_d, "num_clients", "data."), "data.num_clients"),
        num_workers=_as_int(data_d.get("num_workers", 2), "data.num_workers"),
        labels_root=_as_str(data_d.get("labels_root", "labels"), "data.labels_root"),
    )

    pp_d = _require(d, "preprocess", ""); pp_d = pp_d if isinstance(pp_d, dict) else None
    if pp_d is None:
        raise ValueError("preprocess must be a dict")
    input_size = _as_list(_require(pp_d, "input_size", "preprocess."), "preprocess.input_size")
    if len(input_size) != 2:
        raise ValueError("preprocess.input_size must be [H,W]")
    norm_d = _require(pp_d, "normalize", "preprocess."); norm_d = norm_d if isinstance(norm_d, dict) else None
    if norm_d is None:
        raise ValueError("preprocess.normalize must be a dict")
    mean = _as_list(_require(norm_d, "mean", "preprocess.normalize."), "preprocess.normalize.mean")
    std = _as_list(_require(norm_d, "std", "preprocess.normalize."), "preprocess.normalize.std")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("preprocess.normalize.mean/std must be len=3")
    preprocess = PreprocessConfig(
        input_size=(int(input_size[0]), int(input_size[1])),
        normalize_mean=(float(mean[0]), float(mean[1]), float(mean[2])),
        normalize_std=(float(std[0]), float(std[1]), float(std[2])),
        letterbox=_as_bool(pp_d.get("letterbox", True), "preprocess.letterbox"),
    )

    aug_d = _require(d, "augment", ""); aug_d = aug_d if isinstance(aug_d, dict) else None
    if aug_d is None:
        raise ValueError("augment must be a dict")
    ra_d = aug_d.get("random_affine", {}) or {}
    cj_d = aug_d.get("color_jitter", {}) or {}
    mosaic_d = aug_d.get("mosaic", {}) or {}
    mixup_d = aug_d.get("mixup", {}) or {}
    augment = AugmentConfig(
        enabled=_as_bool(_require(aug_d, "enabled", "augment."), "augment.enabled"),
        mosaic_p=_as_float(mosaic_d.get("p", 0.0), "augment.mosaic.p"),
        mixup_p=_as_float(mixup_d.get("p", 0.0), "augment.mixup.p"),
        random_affine_p=_as_float(ra_d.get("p", 0.0), "augment.random_affine.p"),
        degrees=_as_float(ra_d.get("degrees", 0.0), "augment.random_affine.degrees"),
        translate=_as_float(ra_d.get("translate", 0.0), "augment.random_affine.translate"),
        scale=_as_float(ra_d.get("scale", 0.0), "augment.random_affine.scale"),
        color_jitter_p=_as_float(cj_d.get("p", 0.0), "augment.color_jitter.p"),
    )

    fl_d = _require(d, "fl", ""); fl_d = fl_d if isinstance(fl_d, dict) else None
    if fl_d is None:
        raise ValueError("fl must be a dict")
    opt_d = _require(fl_d, "optimizer", "fl."); opt_d = opt_d if isinstance(opt_d, dict) else None
    if opt_d is None:
        raise ValueError("fl.optimizer must be a dict")
    sch_d = _require(fl_d, "scheduler", "fl."); sch_d = sch_d if isinstance(sch_d, dict) else None
    if sch_d is None:
        raise ValueError("fl.scheduler must be a dict")
    fl = FLConfig(
        algorithm=_as_str(_require(fl_d, "algorithm", "fl."), "fl.algorithm"),
        rounds=_as_int(_require(fl_d, "rounds", "fl."), "fl.rounds"),
        clients_per_round=_as_int(_require(fl_d, "clients_per_round", "fl."), "fl.clients_per_round"),
        local_epochs=_as_int(_require(fl_d, "local_epochs", "fl."), "fl.local_epochs"),
        batch_size=_as_int(_require(fl_d, "batch_size", "fl."), "fl.batch_size"),
        optimizer=OptimConfig(
            name=_as_str(_require(opt_d, "name", "fl.optimizer."), "fl.optimizer.name"),
            lr=_as_float(_require(opt_d, "lr", "fl.optimizer."), "fl.optimizer.lr"),
            momentum=_as_float(opt_d.get("momentum", 0.0), "fl.optimizer.momentum"),
            wd=_as_float(opt_d.get("wd", 0.0), "fl.optimizer.wd"),
        ),
        scheduler=SchedulerConfig(
            name=_as_str(_require(sch_d, "name", "fl.scheduler."), "fl.scheduler.name"),
            warmup_epochs=_as_int(sch_d.get("warmup_epochs", 0), "fl.scheduler.warmup_epochs"),
        ),
        clip_norm=_as_float(fl_d.get("clip_norm", 0.0), "fl.clip_norm"),
    )

    model_d = _require(d, "model", ""); model_d = model_d if isinstance(model_d, dict) else None
    if model_d is None:
        raise ValueError("model must be a dict")
    model = ModelConfig(
        type=_as_str(_require(model_d, "type", "model."), "model.type"),
        variant=_as_str(_require(model_d, "variant", "model."), "model.variant"),
        pretrained=_as_bool(_require(model_d, "pretrained", "model."), "model.pretrained"),
        nc=_as_int(_require(model_d, "nc", "model."), "model.nc"),
    )

    attack_d = _require(d, "attack", ""); attack_d = attack_d if isinstance(attack_d, dict) else None
    if attack_d is None:
        raise ValueError("attack must be a dict")
    attack = AttackConfig(
        enabled=_as_bool(_require(attack_d, "enabled", "attack."), "attack.enabled"),
        attacker_clients=[int(x) for x in attack_d.get("attacker_clients", [])],
        poison_rate=_as_float(attack_d.get("poison_rate", 0.0), "attack.poison_rate"),
        target_class=_as_int(attack_d.get("target_class", 0), "attack.target_class"),
        method=_as_str(attack_d.get("method", "trigger_generation"), "attack.method"),
    )

    trigger_d = _require(d, "trigger", ""); trigger_d = trigger_d if isinstance(trigger_d, dict) else None
    if trigger_d is None:
        raise ValueError("trigger must be a dict")
    trigger = TriggerConfig(
        pattern=_as_str(_require(trigger_d, "pattern", "trigger."), "trigger.pattern"),
        size_px=_as_int(_require(trigger_d, "size_px", "trigger."), "trigger.size_px"),
        alpha=_as_float(_require(trigger_d, "alpha", "trigger."), "trigger.alpha"),
        position=_as_str(_require(trigger_d, "position", "trigger."), "trigger.position"),
        apply_prob=_as_float(trigger_d.get("apply_prob", 1.0), "trigger.apply_prob"),
    )

    eval_d = _require(d, "eval", ""); eval_d = eval_d if isinstance(eval_d, dict) else None
    if eval_d is None:
        raise ValueError("eval must be a dict")
    eval_cfg = EvalConfig(
        conf_thresh=_as_float(_require(eval_d, "conf_thresh", "eval."), "eval.conf_thresh"),
        nms_iou=_as_float(_require(eval_d, "nms_iou", "eval."), "eval.nms_iou"),
        max_det=_as_int(_require(eval_d, "max_det", "eval."), "eval.max_det"),
        iou_match=_as_float(_require(eval_d, "iou_match", "eval."), "eval.iou_match"),
        asr_defs=eval_d.get("asr_defs", {}) or {},
    )

    return RootConfig(
        exp=exp,
        data=data,
        preprocess=preprocess,
        augment=augment,
        fl=fl,
        model=model,
        attack=attack,
        trigger=trigger,
        eval=eval_cfg,
    )
