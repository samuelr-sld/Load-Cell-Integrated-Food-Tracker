"""Bounded in-memory result store. Nothing is written to disk; everything vanishes on restart."""
from __future__ import annotations

import threading
from collections import OrderedDict

from .records import InferenceRecord


class ResultTooLarge(Exception):
    pass


class ResultStore:
    def __init__(self, max_results: int, max_bytes: int):
        self.max_results, self.max_bytes = max_results, max_bytes
        self._items: "OrderedDict[str, InferenceRecord]" = OrderedDict()
        self._bytes = 0
        self.evicted_total = 0
        self.lock = threading.RLock()   # also taken by callers that mutate feedback on a fetched record

    def add(self, record: InferenceRecord) -> list[str]:
        """Insert a record, evicting the oldest ones until both limits hold. Returns evicted result IDs."""
        size = record.size_bytes()
        if size > self.max_bytes:
            raise ResultTooLarge(f"result needs {size / 2**20:.1f} MB but the store limit is {self.max_bytes / 2**20:.0f} MB")
        with self.lock:
            evicted: list[str] = []
            while self._items and (len(self._items) >= self.max_results or self._bytes + size > self.max_bytes):
                old_id, old = self._items.popitem(last=False)
                self._bytes -= old.size_bytes()
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
            n = len(self._items)
            self._items.clear()
            self._bytes = 0
            return n

    def stats(self) -> dict:
        with self.lock:
            return {"count": len(self._items), "bytes": self._bytes, "max_results": self.max_results,
                    "max_bytes": self.max_bytes, "evicted_total": self.evicted_total}
