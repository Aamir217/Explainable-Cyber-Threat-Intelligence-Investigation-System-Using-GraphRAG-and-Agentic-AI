"""Rerank fused candidates against the original query for final evidence selection."""

from __future__ import annotations

from cti_graphrag.embeddings import Embedder, cosine_similarity, tokenize
from cti_graphrag.retrieval.vector_store import ScoredChunk


class Reranker:
    """Rescores candidates with a fresh query/document embedding comparison plus a
    lexical-overlap bonus, approximating what a cross-encoder reranker provides
    without requiring a heavyweight model download.
    """

    def __init__(self, embedder: Embedder, lexical_weight: float = 0.3):
        self._embedder = embedder
        self._lexical_weight = lexical_weight

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int = 5) -> list[ScoredChunk]:
        if not candidates:
            return []

        query_vec = self._embedder.embed_one(query)
        query_tokens = set(tokenize(query))
        doc_vecs = self._embedder.embed([c.chunk.text for c in candidates])
        sims = cosine_similarity(query_vec, doc_vecs)

        rescored: list[ScoredChunk] = []
        for candidate, sim in zip(candidates, sims):
            doc_tokens = set(tokenize(candidate.chunk.text))
            overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
            final_score = (1 - self._lexical_weight) * float(sim) + self._lexical_weight * overlap
            rescored.append(ScoredChunk(chunk=candidate.chunk, score=final_score, method=f"reranked({candidate.method})"))

        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return rescored[:top_k]
