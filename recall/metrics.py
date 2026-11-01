"""Thread-safe request metrics for the dashboard."""
from __future__ import annotations

import threading
import time
from collections import deque


class Metrics:
    def __init__(self, window: int = 60):
        self._lock = threading.Lock()
        self.started = time.time()
        self.counts = {"cache": 0, "llm": 0, "no_context": 0, "blocked": 0, "error": 0}
        self.tokens_saved = 0
        self.tokens_spent = 0
        self._lat = {"cache": [], "llm": []}
        self.recent: deque[dict] = deque(maxlen=window)

    def record(self, source: str, total_ms: float, tokens_saved: int = 0, tokens_spent: int = 0, question: str = ""):
        with self._lock:
            self.counts[source] = self.counts.get(source, 0) + 1
            self.tokens_saved += tokens_saved
            self.tokens_spent += tokens_spent
            if source in self._lat:
                self._lat[source].append(total_ms)
                del self._lat[source][:-500]
            self.recent.append({"t": time.time(), "source": source, "ms": round(total_ms, 2), "q": question[:80]})

    @staticmethod
    def _avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    def snapshot(self) -> dict:
        with self._lock:
            answered = self.counts["cache"] + self.counts["llm"]
            return {
                "uptime_s": round(time.time() - self.started),
                "requests": sum(self.counts.values()),
                "counts": dict(self.counts),
                "hit_rate": round(self.counts["cache"] / answered, 4) if answered else 0.0,
                "llm_calls_avoided": self.counts["cache"],
                "tokens_saved": self.tokens_saved,
                "tokens_spent": self.tokens_spent,
                "avg_ms_cache": self._avg(self._lat["cache"]),
                "avg_ms_llm": self._avg(self._lat["llm"]),
                "recent": list(self.recent),
            }
