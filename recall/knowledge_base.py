"""Document store + chunk-level vector retrieval."""
from __future__ import annotations

import hashlib
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chunker import chunk_text
from .embeddings import Embedder
from .vectorstore import FlatIPIndex


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    index: int
    text: str


@dataclass
class Document:
    id: str
    title: str
    text: str
    content_hash: str
    n_chunks: int = 0


@dataclass
class Hit:
    chunk: Chunk
    score: float


class KnowledgeBase:
    def __init__(self, embedder: Embedder, chunk_chars: int = 420):
        self.embedder = embedder
        self.chunk_chars = chunk_chars
        self._lock = threading.RLock()
        self.docs: dict[str, Document] = {}
        self._chunks: list[Chunk] = []
        self._index = FlatIPIndex(embedder.dim)

    # -- mutation -----------------------------------------------------------------
    def add_document(self, title: str, text: str, doc_id: str | None = None) -> tuple[Document, np.ndarray]:
        """Add or replace a document. Returns (doc, chunk_vectors) so callers can
        invalidate cache entries that the new content may affect."""
        title, text = title.strip() or "Untitled", text.strip()
        doc_id = doc_id or _slug(title) or uuid.uuid4().hex[:8]
        pieces = chunk_text(text, self.chunk_chars) or [text]
        vecs = self.embedder.encode([f"{title}. {p}" for p in pieces])
        with self._lock:
            if doc_id in self.docs:
                self._remove_locked(doc_id)
            doc = Document(doc_id, title, text, hashlib.sha1(text.encode()).hexdigest()[:12], len(pieces))
            self.docs[doc_id] = doc
            self._chunks.extend(Chunk(doc_id, title, i, p) for i, p in enumerate(pieces))
            self._index.add(vecs)
        return doc, vecs

    def remove_document(self, doc_id: str) -> bool:
        with self._lock:
            return self._remove_locked(doc_id)

    def _remove_locked(self, doc_id: str) -> bool:
        if doc_id not in self.docs:
            return False
        mask = [c.doc_id != doc_id for c in self._chunks]
        self._chunks = [c for c, keep in zip(self._chunks, mask) if keep]
        self._index.keep(mask)
        del self.docs[doc_id]
        return True

    def load_dir(self, path: Path) -> int:
        n = 0
        for f in sorted(Path(path).glob("*.md")) + sorted(Path(path).glob("*.txt")):
            raw = f.read_text(encoding="utf-8")
            m = re.match(r"\s*#\s+(.+)", raw)
            body = raw[m.end():] if m else raw  # heading becomes the title, not body text
            self.add_document(m.group(1).strip() if m else f.stem, body, doc_id=_slug(f.stem))
            n += 1
        return n

    # -- retrieval ----------------------------------------------------------------
    def search(self, qvec: np.ndarray, k: int = 3) -> list[Hit]:
        with self._lock:
            return [Hit(self._chunks[i], s) for i, s in self._index.search(qvec, k)]

    @property
    def version(self) -> str:
        with self._lock:
            h = hashlib.sha1("|".join(f"{d.id}:{d.content_hash}" for d in sorted(self.docs.values(), key=lambda d: d.id)).encode())
            return h.hexdigest()[:10]

    def listing(self) -> list[dict]:
        with self._lock:
            return [
                {"id": d.id, "title": d.title, "chunks": d.n_chunks, "chars": len(d.text), "hash": d.content_hash}
                for d in self.docs.values()
            ]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]
