"""Sentence-aware chunking with one-sentence overlap."""
from __future__ import annotations

import re

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for block in re.split(r"\n\s*\n|\n(?=#)", text):
        block = re.sub(r"^#+\s*", "", block.strip())
        block = re.sub(r"\s+", " ", block)
        if block:
            out.extend(s.strip() for s in _SENT.split(block) if s.strip())
    return out


def chunk_text(text: str, max_chars: int = 420, overlap_sentences: int = 1) -> list[str]:
    sents = split_sentences(text)
    chunks: list[str] = []
    cur: list[str] = []
    for s in sents:
        if cur and sum(len(x) + 1 for x in cur) + len(s) > max_chars:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
        cur.append(s)
    if cur and (not chunks or " ".join(cur) != chunks[-1]):
        chunks.append(" ".join(cur))
    return chunks
