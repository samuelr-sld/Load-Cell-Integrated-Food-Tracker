"""Stage timing with CUDA synchronization so GPU work is fully counted inside each timed section."""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch


def synchronize(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


class StageTimer:
    def __init__(self, device: torch.device | None):
        self.device = device
        self.ms: dict[str, float] = {}
        synchronize(device)
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        synchronize(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            synchronize(self.device)
            self.ms[name] = self.ms.get(name, 0.0) + (time.perf_counter() - start) * 1000.0

    def total_ms(self) -> float:
        synchronize(self.device)
        return (time.perf_counter() - self._t0) * 1000.0
