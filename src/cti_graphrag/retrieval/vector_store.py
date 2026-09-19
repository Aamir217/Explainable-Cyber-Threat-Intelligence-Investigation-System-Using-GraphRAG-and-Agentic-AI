"""In-memory dense vector index over document chunks (cosine similarity)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cti_graphrag.embeddings import Embedder, cosine_similarity
from cti_graphrag.retrieval.chunking import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    method: str = "vector"


class VectorStore:
    def __init__(self, embedder: Embedder):
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._matrix = self._embedder.embed([c.text for c in chunks]) if chunks else None

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        if self._matrix is None or len(self._chunks) == 0:
            return []
        query_vec = self._embedder.embed_one(query)
        sims = cosine_similarity(query_vec, self._matrix)
        top_idx = np.argsort(-sims)[:top_k]
        return [ScoredChunk(chunk=self._chunks[i], score=float(sims[i]), method="vector") for i in top_idx]
