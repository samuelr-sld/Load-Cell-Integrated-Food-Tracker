"""Bounded in-memory result store. Nothing is written to disk; everything vanishes on restart."""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Callable

from .records import InferenceRecord


log = logging.getLogger("testbench.store")


class ResultTooLarge(Exception):
    pass


class ResultStore:
    def __init__(self, max_results: int, max_bytes: int):
        self.max_results, self.max_bytes = max_results, max_bytes
        self._items: "OrderedDict[str, InferenceRecord]" = OrderedDict()
        self._bytes = 0
        self.evicted_total = 0
        self.protected: Callable[[], set] = set      # result ids that must outlive "Clear all" and eviction (results of unacknowledged scans)
        self.lock = threading.RLock()   # also taken by callers that mutate feedback on a fetched record

    def add(self, record: InferenceRecord) -> list[str]:
        """Insert a record, evicting the oldest ones until both limits hold. Returns evicted result IDs."""
        size = record.size_bytes()
        if size > self.max_bytes:
            raise ResultTooLarge(f"result needs {size / 2**20:.1f} MB but the store limit is {self.max_bytes / 2**20:.0f} MB")
        with self.lock:
            evicted: list[str] = []
            keep = self.protected()
            while len(self._items) >= self.max_results or self._bytes + size > self.max_bytes:
                # shortcut: a protected result is never evicted, so the store may exceed its limits by the (<=50) pending scans
                old_id = next((i for i in self._items if i not in keep), None)
                if old_id is None:
                    break
                self._bytes -= self._items.pop(old_id).size_bytes()
                evicted.append(old_id)
            self._items[record.result_id] = record
            self._bytes += size
            self.evicted_total += len(evicted)
            return evicted

    def get(self, result_id: str) -> InferenceRecord:
        with self.lock:
            return self._items[result_id]   # KeyError -> caller maps to 404

    def list(self) -> list[InferenceRecord]:
        """Newest first."""
        with self.lock:
            return list(reversed(self._items.values()))

    def clear(self) -> int:
        with self.lock:
            keep = self.protected()
            gone = [i for i in self._items if i not in keep]
            for i in gone:
                self._bytes -= self._items.pop(i).size_bytes()
            log.info("results cleared: %d removed, %d kept for pending scans", len(gone), len(self._items))
            return len(gone)

    def stats(self) -> dict:
        with self.lock:
            return {"count": len(self._items), "bytes": self._bytes, "max_results": self.max_results,
                    "max_bytes": self.max_bytes, "evicted_total": self.evicted_total}
