# FL Detection Backdoor/Poisoning Baseline (MVP)

Goal: swap detection models (YOLO3/5/8, Faster R-CNN, etc.) while keeping **partitioning / preprocessing / evaluation / NMS / conf-thresh / ASR / seed** fixed in common code.

This repository is a **minimal runnable baseline**:
- `flwr` FedAvg wiring included (simulation mode).
- A simple toy detector is included so you can run end-to-end without external detection frameworks.
- Real model adapters (ultralytics / detectron2) are intentionally left as TODOs so developers only add files under `src/models/`.

## Run

```bash
python -m src.main --config configs/mvp_synth_yolo_toy_fedavg.yaml
```

Outputs are written to `logs/{model_type}_{variant}_{timestamp}/`:
- `epoch_*.jsonl` (per local epoch metrics for a sample client in MVP)
- `round_*.jsonl` (per FL round global eval metrics)
- `best.pt`, `last.pt` (global model checkpoints)
- `config_resolved.yaml`, `meta.json`

## Dataset Assumptions (MVP)

- For real datasets (LISA/BDD100K), this baseline expects YOLO-format labels:
  - image path list in `splits/*.txt`, one image path per line
  - corresponding label file per image: same path with extension `.txt` under a `labels/` root (configurable)
- If `data.dataset_id: synthetic`, a small synthetic detection dataset is generated in-memory for smoke tests.

## Developer Extension Points

- Add a model adapter: create `src/models/<your_adapter>.py` and register it in `src/models/registry.py`.
- Add an attack: create `src/attacks/<your_attack>.py` and register it in `src/attacks/registry.py`.

