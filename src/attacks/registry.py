from __future__ import annotations

from typing import Callable, Dict, Optional

from ..config.schema import AttackConfig, EvalConfig
from .base import Attack
from .dba_skeleton import DBASkeletonAttack
from .trigger_attacks import (
    TriggerDisappearanceAttack,
    TriggerGenerationAttack,
    TriggerGlobalMisclsAttack,
    TriggerRegionalMisclsAttack,
)


def make_attack(attack_cfg: AttackConfig, eval_cfg: EvalConfig, *, client_id: int) -> Optional[Attack]:
    if not attack_cfg.enabled:
        return None

    m = attack_cfg.method
    if m == "trigger_generation":
        return TriggerGenerationAttack(target_class=attack_cfg.target_class)
    if m == "trigger_regional_miscls":
        roi_expand_px = int(eval_cfg.asr_defs.get("regional", {}).get("roi_expand_px", 50))
        return TriggerRegionalMisclsAttack(target_class=attack_cfg.target_class, roi_expand_px=roi_expand_px)
    if m == "trigger_global_miscls":
        return TriggerGlobalMisclsAttack(target_class=attack_cfg.target_class)
    if m == "trigger_disappearance":
        roi_expand_px = int(eval_cfg.asr_defs.get("regional", {}).get("roi_expand_px", 50))
        return TriggerDisappearanceAttack(target_class=attack_cfg.target_class, roi_expand_px=roi_expand_px)
    if m == "dba":
        return DBASkeletonAttack(target_class=attack_cfg.target_class, client_id=client_id)
    raise ValueError(f"Unknown attack method: {m}")

