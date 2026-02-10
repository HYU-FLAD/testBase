from __future__ import annotations

import os
from typing import List, Tuple

from ..config.schema import DataConfig, ModelConfig, PreprocessConfig
from .class_mapping import get_class_remap
from .datasets.synthetic import SyntheticDetDataset
from .datasets.yolo_txt import YoloTxtDetDataset


def make_dataset(
    *,
    dataset_id: str,
    keys: List[str],
    data_cfg: DataConfig,
    preprocess_cfg: PreprocessConfig,
    model_cfg: ModelConfig,
    seed: int,
):
    dataset_id = dataset_id.lower()
    if dataset_id == "synthetic":
        return SyntheticDetDataset(
            keys,
            seed=seed,
            input_size=preprocess_cfg.input_size,
            letterbox_enabled=preprocess_cfg.letterbox,
            mean=preprocess_cfg.normalize_mean,
            std=preprocess_cfg.normalize_std,
            nc=model_cfg.nc,
        )

    # MVP assumption: real datasets are prepared as YOLO txt with unified class ids already.
    root = os.path.abspath(os.path.join(data_cfg.root, dataset_id))
    remap = get_class_remap(dataset_id)
    return YoloTxtDetDataset(
        keys,
        root=root,
        labels_root=data_cfg.labels_root,
        class_remap=remap,
        input_size=preprocess_cfg.input_size,
        letterbox_enabled=preprocess_cfg.letterbox,
        mean=preprocess_cfg.normalize_mean,
        std=preprocess_cfg.normalize_std,
        nc=model_cfg.nc,
    )
