"""Sparse lexical retrieval over document chunks using BM25."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from cti_graphrag.embeddings import tokenize
from cti_graphrag.retrieval.chunking import Chunk
from cti_graphrag.retrieval.vector_store import ScoredChunk


class BM25Store:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        if chunks:
            self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [ScoredChunk(chunk=self._chunks[i], score=float(scores[i]), method="bm25") for i in ranked if scores[i] > 0]
