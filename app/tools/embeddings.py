"""Embedders for vector evidence scoring.

Provides:
- HashingEmbedder: fast, deterministic, offline bag-of-words/n-gram hashing (0 download)
- SentenceTransformerEmbedder: dense embeddings using BGE/MiniLM when available
"""

from __future__ import annotations

import re
import zlib
from typing import Protocol

import numpy as np


def tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens."""
    return [t for t in re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower()) if len(t) > 1]


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1.0, norms)).astype(np.float32)


class HashingEmbedder:
    """Deterministic signed bag-of-(uni+bi)grams. Offline, instant, and reliable."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        if not toks:
            return v
        for t in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]:
            h = zlib.crc32(t.encode())
            v[h % self.dim] += 1.0 if (zlib.crc32(t.encode() + b"#") & 1) else -1.0
        norm = np.linalg.norm(v)
        return (v / norm).astype(np.float32) if norm > 0 else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._vec(text)
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32) if norm > 0 else vec


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=32), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)


def get_embedder(embedder_type: str = "hash") -> Embedder:
    if embedder_type == "bge":
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            return HashingEmbedder()
    return HashingEmbedder()

