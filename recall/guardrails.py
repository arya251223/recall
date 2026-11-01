"""Input / output safety checks and a per-client rate limiter."""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

from .embeddings import content_tokens

MAX_QUERY_CHARS = 500
MIN_QUERY_CHARS = 3

_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any |the )?(previous|prior|above|earlier|your) (instructions|prompts?|rules)",
    r"(reveal|show|print|repeat|leak) (me )?(your|the) (system )?(prompt|instructions)",
    r"you are now (?!able)",
    r"\bjailbreak\b",
    r"developer mode",
    r"act as (an? )?(unrestricted|unfiltered|dan)\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION), re.I)

_REDACTIONS = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("api_key", re.compile(r"\b(AIza[0-9A-Za-z_\-]{30,}|sk-[A-Za-z0-9]{20,})\b")),
    ("phone", re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)")),
]


@dataclass
class GuardResult:
    ok: bool
    reason: str = ""


def check_input(query: str) -> GuardResult:
    q = query.strip()
    if len(q) < MIN_QUERY_CHARS:
        return GuardResult(False, "Question is too short.")
    if len(q) > MAX_QUERY_CHARS:
        return GuardResult(False, f"Question is longer than {MAX_QUERY_CHARS} characters.")
    if _INJECTION_RE.search(q):
        return GuardResult(False, "Blocked: the question looks like a prompt-injection attempt.")
    return GuardResult(True)


def sanitize_output(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    for label, rx in _REDACTIONS:
        if rx.search(text):
            found.append(label)
            text = rx.sub(f"[redacted {label}]", text)
    return text, found


def grounding_score(answer: str, context: str) -> float:
    """Fraction of the answer's content tokens that also occur in the retrieved context."""
    a = set(content_tokens(answer))
    if not a:
        return 1.0
    return len(a & set(content_tokens(context))) / len(a)


class RateLimiter:
    """Token bucket per client key."""

    def __init__(self, per_minute: int):
        self.capacity = max(1, per_minute)
        self.rate = self.capacity / 60.0
        self._b: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._b.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1:
                self._b[key] = (tokens, now)
                return False
            self._b[key] = (tokens - 1, now)
            return True
