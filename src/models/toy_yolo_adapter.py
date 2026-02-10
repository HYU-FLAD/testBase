from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelAdapter


def _boxes_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = torch.maximum(ax1, bx1)
    inter_y1 = torch.maximum(ay1, by1)
    inter_x2 = torch.minimum(ax2, bx2)
    inter_y2 = torch.minimum(ay2, by2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    return inter / (area_a + area_b - inter + 1e-9)


class ToyDetNet(nn.Module):
    def __init__(self, *, nc: int, k: int = 50) -> None:
        super().__init__()
        self.nc = int(nc)
        self.k = int(k)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, self.k * (4 + 1 + self.nc))  # box + obj + cls logits

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        f = self.backbone(x).view(b, -1)
        y = self.fc(f).view(b, self.k, 4 + 1 + self.nc)
        box_raw = y[:, :, :4]
        obj_raw = y[:, :, 4]
        cls_logits = y[:, :, 5:]
        box = torch.sigmoid(box_raw)  # 0..1
        # force x2>=x1, y2>=y1
        x1y1 = torch.minimum(box[:, :, 0:2], box[:, :, 2:4])
        x2y2 = torch.maximum(box[:, :, 0:2], box[:, :, 2:4])
        box = torch.cat([x1y1, x2y2], dim=-1)
        return box, obj_raw, cls_logits


@dataclass
class ToyYoloAdapter(ModelAdapter):
    nc: int
    device: torch.device
    input_size: Tuple[int, int]
    optimizer_cfg: Dict[str, Any]
    clip_norm: float = 0.0

    def __post_init__(self) -> None:
        self.model = ToyDetNet(nc=self.nc, k=50).to(self.device)
        self._param_keys = sorted(self.model.state_dict().keys())
        self._build_optim()

    def _build_optim(self) -> None:
        name = str(self.optimizer_cfg.get("name", "sgd")).lower()
        lr = float(self.optimizer_cfg.get("lr", 1e-2))
        wd = float(self.optimizer_cfg.get("wd", 0.0))
        if name == "sgd":
            mom = float(self.optimizer_cfg.get("momentum", 0.9))
            self.optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
        elif name == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {name}")

    def train_one_epoch(self, train_loader) -> Dict[str, float]:
        self.model.train()
        tot_loss = 0.0
        tot_box = 0.0
        tot_obj = 0.0
        tot_cls = 0.0
        steps = 0
        h, w = self.input_size
        for images, targets in train_loader:
            x = torch.stack([im.to(self.device) for im in images], dim=0)
            # targets: list[dict]
            gt_boxes = [t["boxes"].to(self.device) for t in targets]
            gt_labels = [t["labels"].to(self.device) for t in targets]

            pred_box_n, pred_obj_raw, pred_cls_logits = self.model(x)
            # convert normalized boxes to absolute for loss matching
            pred_box = pred_box_n.clone()
            pred_box[:, :, [0, 2]] *= float(w)
            pred_box[:, :, [1, 3]] *= float(h)

            loss_box = torch.tensor(0.0, device=self.device)
            loss_obj = torch.tensor(0.0, device=self.device)
            loss_cls = torch.tensor(0.0, device=self.device)

            bsz, k = pred_box.shape[0], pred_box.shape[1]
            obj_target = torch.zeros((bsz, k), device=self.device)
            cls_target = torch.full((bsz, k), -1, device=self.device, dtype=torch.long)
            box_target = torch.zeros((bsz, k, 4), device=self.device)

            for bi in range(bsz):
                if gt_boxes[bi].numel() == 0:
                    continue
                iou = _boxes_iou_xyxy(pred_box[bi], gt_boxes[bi])  # [K,Ng]
                # greedy: for each gt, pick best remaining pred
                used = torch.zeros((k,), dtype=torch.bool, device=self.device)
                for gi in range(gt_boxes[bi].shape[0]):
                    scores = iou[:, gi].clone()
                    scores[used] = -1.0
                    pi = int(torch.argmax(scores).item())
                    if scores[pi].item() < 0:
                        continue
                    used[pi] = True
                    obj_target[bi, pi] = 1.0
                    cls_target[bi, pi] = gt_labels[bi][gi]
                    box_target[bi, pi] = gt_boxes[bi][gi]

            # losses
            loss_obj = F.binary_cross_entropy_with_logits(pred_obj_raw, obj_target, reduction="mean")
            m_pos = cls_target >= 0
            if m_pos.any():
                loss_cls = F.cross_entropy(pred_cls_logits[m_pos], cls_target[m_pos], reduction="mean")
                loss_box = F.smooth_l1_loss(pred_box[m_pos], box_target[m_pos], reduction="mean")
            loss = loss_obj + loss_cls + loss_box

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            if float(self.clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.clip_norm))
            self.optim.step()

            tot_loss += float(loss.item())
            tot_box += float(loss_box.item()) if torch.is_tensor(loss_box) else float(loss_box)
            tot_obj += float(loss_obj.item()) if torch.is_tensor(loss_obj) else float(loss_obj)
            tot_cls += float(loss_cls.item()) if torch.is_tensor(loss_cls) else float(loss_cls)
            steps += 1

        denom = max(1, steps)
        return {
            "loss": tot_loss / denom,
            "loss_box": tot_box / denom,
            "loss_obj": tot_obj / denom,
            "loss_cls": tot_cls / denom,
        }

    @torch.no_grad()
    def predict(self, images) -> List[np.ndarray]:
        self.model.eval()
        h, w = self.input_size
        x = torch.stack([im.to(self.device) for im in images], dim=0)
        box_n, obj_raw, cls_logits = self.model(x)
        obj = torch.sigmoid(obj_raw)  # [B,K]
        cls_prob = torch.softmax(cls_logits, dim=-1)  # [B,K,C]
        cls_score, cls_id = torch.max(cls_prob, dim=-1)  # [B,K]
        score = obj * cls_score

        box = box_n.clone()
        box[:, :, [0, 2]] *= float(w)
        box[:, :, [1, 3]] *= float(h)

        out: List[np.ndarray] = []
        for bi in range(box.shape[0]):
            det = torch.cat(
                [
                    box[bi],
                    score[bi].unsqueeze(-1),
                    cls_id[bi].float().unsqueeze(-1),
                ],
                dim=-1,
            )  # [K,6]
            out.append(det.detach().cpu().numpy().astype(np.float32))
        return out

    def get_weights(self) -> List[np.ndarray]:
        sd = self.model.state_dict()
        return [sd[k].detach().cpu().numpy() for k in self._param_keys]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        sd = self.model.state_dict()
        if len(weights) != len(self._param_keys):
            raise ValueError(f"Weight length mismatch: {len(weights)} vs {len(self._param_keys)}")
        for k, w in zip(self._param_keys, weights):
            t = torch.from_numpy(np.asarray(w)).to(sd[k].device)
            if sd[k].shape != t.shape:
                raise ValueError(f"Shape mismatch for {k}: {sd[k].shape} vs {t.shape}")
            sd[k].copy_(t)

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "nc": self.nc,
                "input_size": self.input_size,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])

