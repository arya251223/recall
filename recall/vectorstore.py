"""Exact inner-product vector index (equivalent to ``faiss.IndexFlatIP``)."""
from __future__ import annotations

import numpy as np


class FlatIPIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._mat = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return self._mat.shape[0]

    def add(self, vecs: np.ndarray) -> None:
        vecs = np.atleast_2d(vecs).astype(np.float32, copy=False)
        self._mat = np.vstack([self._mat, vecs])

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if len(self) == 0 or k <= 0:
            return []
        sims = self._mat @ np.asarray(query, dtype=np.float32).reshape(-1)
        k = min(k, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    def keep(self, mask: list[bool]) -> None:
        """Retain only rows where mask is True (used for deletions)."""
        self._mat = self._mat[np.asarray(mask, dtype=bool)]

    def max_similarity(self, queries: np.ndarray) -> np.ndarray:
        """For each stored row, max cosine against the given query vectors."""
        if len(self) == 0 or len(queries) == 0:
            return np.zeros(len(self), dtype=np.float32)
        return (self._mat @ np.atleast_2d(queries).T).max(axis=1)
