"""Thread-safe in-memory vector store for chunk-level evidence search."""

from __future__ import annotations

import threading

import numpy as np

from app.tools.chunking import DocumentChunk
from app.tools.embeddings import Embedder, get_embedder


class VectorStore:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or get_embedder("hash")
        self._chunks: dict[str, DocumentChunk] = {}
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, chunks: list[DocumentChunk]) -> int:
        """Embeds and indexes new chunks. Skips existing chunk_ids."""
        with self._lock:
            new = [c for c in chunks if c.chunk_id not in self._chunks]
            if not new:
                return 0
            vecs = self.embedder.embed_documents([c.text for c in new])
            self._mat = vecs if self._mat is None else np.vstack([self._mat, vecs])
            for c in new:
                self._chunks[c.chunk_id] = c
                self._ids.append(c.chunk_id)
            return len(new)

    def get(self, chunk_id: str) -> DocumentChunk | None:
        return self._chunks.get(chunk_id)

    def search(
        self,
        query: str,
        k: int = 5,
        doc_ids: set[str] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Returns top-k chunks scored by cosine similarity."""
        with self._lock:
            if self._mat is None or not self._ids:
                return []
            q_vec = self.embedder.embed_query(query)
            scores = self._mat @ q_vec
            if doc_ids is not None:
                mask = np.array([self._chunks[cid].doc_id in doc_ids for cid in self._ids])
                scores = np.where(mask, scores, -np.inf)
            order = np.argsort(-scores)[:k]
            results: list[tuple[DocumentChunk, float]] = []
            for i in order:
                s = float(scores[i])
                if np.isfinite(s) and s > 0.05:
                    results.append((self._chunks[self._ids[i]], s))
            return results


_GLOBAL_VECTOR_STORE: VectorStore | None = None
_STORE_LOCK = threading.Lock()


def get_global_vector_store() -> VectorStore:
    global _GLOBAL_VECTOR_STORE
    if _GLOBAL_VECTOR_STORE is None:
        with _STORE_LOCK:
            if _GLOBAL_VECTOR_STORE is None:
                _GLOBAL_VECTOR_STORE = VectorStore()
    return _GLOBAL_VECTOR_STORE
