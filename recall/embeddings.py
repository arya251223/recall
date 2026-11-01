"""Embedding backends.

* ``HashingEmbedder``  - dependency-free, deterministic, lexical-semantic (stemming +
  synonym canonicalisation + hashed n-grams).  Used offline and in CI.
* ``SentenceTransformerEmbedder`` - sentence-transformers/all-MiniLM-L6-v2 (true dense
  semantics).  Auto-selected when the library and model are available.

Every embedder returns L2-normalised float32 vectors, so inner product == cosine.
Similarity thresholds are *embedder specific*, hence the ``default_*`` attributes.
"""
from __future__ import annotations

import re
import zlib
from functools import lru_cache

import numpy as np

STOPWORDS = frozenset(
    """a an the is are was were be been being of to in on at for from by with and or but if
    then so do does did can could should would will shall may might i you we they he she it my
    your our their me us this that these those what which who whom how when where why there
    here as about into than too very just please tell explain give get need want""".split()
)

# Small domain-agnostic synonym table -> canonical token.
_SYNONYMS = {
    **dict.fromkeys(
        ["price", "prices", "pricing", "cost", "costs", "fee", "fees", "charge", "charges",
         "charged", "pay", "paying", "payment", "expensive", "cheap"], "price"),
    **dict.fromkeys(["refund", "refunds", "refunded", "reimburse", "reimbursed", "reimbursement",
                     "moneyback"], "refund"),
    **dict.fromkeys(["cancel", "cancels", "cancelling", "canceling", "cancellation", "terminate",
                     "unsubscribe", "quit", "stop"], "cancel"),
    **dict.fromkeys(["limit", "limits", "cap", "caps", "quota", "quotas", "maximum", "max",
                     "throttle", "throttled", "throttling", "restriction", "restrictions", "rate", "rates"], "limit"),
    **dict.fromkeys(["login", "signin", "authenticate", "authentication", "sso", "saml", "oidc",
                     "okta"], "auth"),
    **dict.fromkeys(["backup", "backups", "snapshot", "snapshots", "restore", "recover", "recovery",
                     "disaster"], "backup"),
    **dict.fromkeys(["encrypt", "encrypted", "encryption", "secure", "security", "aes", "tls"], "secure"),
    **dict.fromkeys(["delete", "deleted", "deleting", "deletion", "remove", "erase", "retention",
                     "retain", "retained", "trash", "bin"], "retention"),
    **dict.fromkeys(["support", "helpdesk", "assistance", "contact", "help"], "support"),
    **dict.fromkeys(["share", "sharing", "shared", "permission", "permissions", "access", "role",
                     "roles", "link", "links"], "share"),
    **dict.fromkeys(["region", "regions", "residency", "location", "locations", "datacenter",
                     "datacentre", "geography", "country"], "region"),
    **dict.fromkeys(["sync", "synchronise", "synchronize", "desktop", "client", "app"], "sync"),
    **dict.fromkeys(["import", "migrate", "migration", "move", "transfer", "switch"], "migrate"),
    **dict.fromkeys(["upload", "uploads", "uploading", "file", "files", "size"], "upload"),
    **dict.fromkeys(["plan", "plans", "tier", "tiers", "subscription", "subscriptions"], "plan"),
}


def stem(word: str) -> str:
    if word in _SYNONYMS:
        return _SYNONYMS[word]
    if len(word) <= 3:
        return word
    if word.endswith("ss"):
        return word
    for suf in ("ations", "ation", "ings", "ing", "edly", "ed", "ies", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            base = word[: -len(suf)] + ("y" if suf == "ies" else "")
            return _SYNONYMS.get(base, base)
    return word


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def content_tokens(text: str) -> list[str]:
    """Lower-cased, stop-word-free, stemmed/canonicalised tokens."""
    return [stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@lru_cache(maxsize=200_000)
def _slot(feature: str, dim: int) -> tuple[int, float]:
    h = zlib.crc32(feature.encode("utf-8"))
    return h % dim, 1.0 if (h >> 31) & 1 else -1.0


class Embedder:
    name = "base"
    dim = 0
    default_cache_threshold = 0.85
    default_min_relevance = 0.30
    default_invalidate_similarity = 0.45  # query<->new-chunk similarity that makes a cached answer stale

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class HashingEmbedder(Embedder):
    name = "hashing-lexical"
    default_cache_threshold = 0.35  # tuned on a held-out split, see docs/REPORT.md
    default_min_relevance = 0.12
    default_invalidate_similarity = 0.30

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _features(self, text: str):
        toks = content_tokens(text)
        for t in toks:
            yield "u:" + t, 1.0
            padded = f"^{t}$"
            for i in range(len(padded) - 2):
                yield "c:" + padded[i : i + 3], 0.15
        for a, b in zip(toks, toks[1:]):
            yield f"b:{a}_{b}", 0.5

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, w in self._features(text):
                idx, sign = _slot(feat, self.dim)
                out[row, idx] += sign * w
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder(Embedder):
    name = "all-MiniLM-L6-v2"
    default_cache_threshold = 0.85
    default_min_relevance = 0.30

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def build_embedder(choice: str = "auto") -> Embedder:
    if choice in ("auto", "minilm"):
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:  # library or model weights unavailable
            if choice == "minilm":
                raise RuntimeError(f"MiniLM embedder requested but unavailable: {exc}") from exc
    return HashingEmbedder()
