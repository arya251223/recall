"""Semantic cache with the safeguards a naive "cosine >= threshold" cache lacks.

* TTL + LRU eviction (bounded memory, bounded staleness)
* Targeted invalidation when knowledge-base documents change
* Entity guard   - rejects near-duplicates that differ on a discriminating entity
                   (e.g. "Pro plan price" vs "Business plan price")
* Retrieval-consistency guard - a hit must be supported by the same documents the
                   new question would retrieve
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .embeddings import STOPWORDS, _SYNONYMS, stem
from .vectorstore import FlatIPIndex

CONTRAST_WORDS = frozenset(
    {"monthly", "annual", "annually", "yearly", "weekly", "daily", "hourly", "free", "paid"}
)
_CONTRAST_CANON = {"annually": "annual", "yearly": "annual"}
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-.]*")


def extract_entities(query: str) -> frozenset[str]:
    """Numbers, capitalised proper-noun-like words and contrast qualifiers."""
    ents: set[str] = set()
    for t in _WORD.findall(query):
        tl = t.lower().strip(".-")
        if not tl:
            continue
        if any(c.isdigit() for c in tl) or tl in CONTRAST_WORDS:
            ents.add(_CONTRAST_CANON.get(tl, tl))
        elif t[0].isupper() and tl not in STOPWORDS and tl not in _SYNONYMS:
            ents.add(stem(tl))
    return frozenset(ents)


def entity_conflict(a: frozenset[str], b: frozenset[str]) -> tuple[bool, str]:
    a_only, b_only = a - b, b - a
    if a_only and b_only:
        return True, f"{sorted(a_only)} vs {sorted(b_only)}"
    return False, ""


@dataclass
class CacheEntry:
    id: int
    query: str
    answer: str
    sources: list[dict]
    doc_ids: frozenset[str]
    primary_doc: str | None
    entities: frozenset[str]
    tokens: int
    created: float
    last_access: float
    hits: int = 0


@dataclass
class Lookup:
    hit: bool
    score: float = 0.0
    entry: CacheEntry | None = None
    reason: str = "empty"
    rejected: list[dict] = field(default_factory=list)


class SemanticCache:
    def __init__(self, dim: int, threshold: float = 0.85, ttl_seconds: float = 86_400,
                 max_entries: int = 1_000, entity_guard: bool = True):
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.entity_guard = entity_guard
        self._index = FlatIPIndex(dim)
        self._entries: list[CacheEntry] = []
        self._lock = threading.RLock()
        self._next_id = 1
        self.counters = {"lookups": 0, "hits": 0, "expired": 0, "evicted": 0, "invalidated": 0,
                         "rejected_entity": 0, "rejected_retrieval": 0}

    def __len__(self) -> int:
        return len(self._entries)

    # -- core ---------------------------------------------------------------------
    CONSISTENCY_TOP_N = 2

    def lookup(self, vec: np.ndarray, query: str, retrieved_doc_ids: list[str] | None = None,
               now: float | None = None) -> Lookup:
        """``retrieved_doc_ids``: document ids the *new* question retrieves, best first. When given,
        a hit is only accepted if the cached answer's primary source is among the top N."""
        now = time.time() if now is None else now
        with self._lock:
            self.counters["lookups"] += 1
            self._drop_expired(now)
            cands = self._index.search(vec, 5)
            if not cands:
                return Lookup(False, reason="empty")
            q_ents = extract_entities(query)
            rejected: list[dict] = []
            for idx, score in cands:
                if score < self.threshold:
                    break
                e = self._entries[idx]
                if self.entity_guard:
                    bad, why = entity_conflict(q_ents, e.entities)
                    if bad:
                        self.counters["rejected_entity"] += 1
                        rejected.append({"query": e.query, "score": round(score, 3),
                                         "reason": "entity_conflict", "detail": why})
                        continue
                if (retrieved_doc_ids is not None and e.primary_doc
                        and e.primary_doc not in retrieved_doc_ids[: self.CONSISTENCY_TOP_N]):
                    self.counters["rejected_retrieval"] += 1
                    rejected.append({"query": e.query, "score": round(score, 3),
                                     "reason": "retrieval_mismatch", "detail": "different source documents"})
                    continue
                e.hits += 1
                e.last_access = now
                self.counters["hits"] += 1
                return Lookup(True, score, e, "hit", rejected)
            reason = rejected[0]["reason"] if rejected else "below_threshold"
            return Lookup(False, cands[0][1], None, reason, rejected)

    def add(self, vec: np.ndarray, query: str, answer: str, sources: list[dict], tokens: int,
            now: float | None = None) -> CacheEntry:
        now = time.time() if now is None else now
        with self._lock:
            e = CacheEntry(self._next_id, query, answer, sources, frozenset(s["doc_id"] for s in sources),
                           sources[0]["doc_id"] if sources else None, extract_entities(query), tokens, now, now)
            self._next_id += 1
            self._index.add(vec)
            self._entries.append(e)
            while len(self._entries) > self.max_entries:
                victim = min(range(len(self._entries)), key=lambda i: self._entries[i].last_access)
                self._remove_mask([i != victim for i in range(len(self._entries))])
                self.counters["evicted"] += 1
            return e

    # -- invalidation -------------------------------------------------------------
    def invalidate_docs(self, doc_ids: set[str]) -> int:
        with self._lock:
            return self._drop(lambda e: bool(e.doc_ids & doc_ids))

    def invalidate_similar(self, chunk_vecs: np.ndarray, min_sim: float) -> int:
        """Drop entries whose *question* is close to newly added content (their answer
        may now be incomplete or wrong)."""
        with self._lock:
            sims = self._index.max_similarity(chunk_vecs)
            return self._drop_by_mask([bool(s >= min_sim) for s in sims])

    def clear(self) -> int:
        with self._lock:
            n = len(self._entries)
            self._entries, self._index = [], FlatIPIndex(self._index.dim)
            return n

    # -- helpers ------------------------------------------------------------------
    def _drop_expired(self, now: float) -> None:
        n = self._drop(lambda e: now - e.created > self.ttl)
        self.counters["expired"] += n
        self.counters["invalidated"] -= n  # expiry is not an invalidation

    def _drop(self, pred) -> int:
        return self._drop_by_mask([pred(e) for e in self._entries])

    def _drop_by_mask(self, drop: list[bool]) -> int:
        if not any(drop):
            return 0
        self._remove_mask([not d for d in drop])
        n = sum(drop)
        self.counters["invalidated"] += n
        return n

    def _remove_mask(self, keep: list[bool]) -> None:
        self._index.keep(keep)
        self._entries = [e for e, k in zip(self._entries, keep) if k]

    def view(self, limit: int = 50) -> list[dict]:
        with self._lock:
            es = sorted(self._entries, key=lambda e: -e.last_access)[:limit]
            return [{"id": e.id, "query": e.query, "answer": e.answer, "hits": e.hits,
                     "docs": sorted(e.doc_ids), "tokens": e.tokens,
                     "age_s": round(time.time() - e.created, 1)} for e in es]
