"""MobileNetV3 inference: logits -> softmax -> top-k with the training class order."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Prediction:
    rank: int
    class_index: int
    label: str
    softmax_score: float   # raw softmax output; NOT a calibrated probability


class ClassificationService:
    def __init__(self, model: nn.Module, class_names: list[str], device: torch.device, batch_size: int = 32, top_k: int = 5):
        self.model, self.class_names, self.device = model, class_names, device
        self.batch_size, self.top_k = max(1, batch_size), min(top_k, len(class_names))
        self.model.eval()

    def classify(self, tensors: list[torch.Tensor]) -> list[list[Prediction]]:
        """One prediction list per input tensor, in input order. Eval mode + inference_mode (no gradients)."""
        out: list[list[Prediction]] = []
        with torch.inference_mode():
            for i in range(0, len(tensors), self.batch_size):
                batch = torch.stack(tensors[i:i + self.batch_size]).to(self.device, non_blocking=True)
                logits = self.model(batch)
                if logits.shape[1] != len(self.class_names):
                    raise RuntimeError(f"classifier emits {logits.shape[1]} logits but {len(self.class_names)} class names are loaded")
                scores, idx = torch.softmax(logits.float(), dim=1).topk(self.top_k, dim=1)
                for s_row, i_row in zip(scores.cpu().tolist(), idx.cpu().tolist()):
                    out.append([Prediction(r + 1, ci, self.class_names[ci], float(s)) for r, (s, ci) in enumerate(zip(s_row, i_row))])
        return out
