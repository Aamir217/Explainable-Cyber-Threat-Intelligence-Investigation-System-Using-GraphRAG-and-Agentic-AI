"""Fuse ranked results from multiple retrievers (dense, sparse, graph) into one ranking."""

from __future__ import annotations

from cti_graphrag.retrieval.vector_store import ScoredChunk


def reciprocal_rank_fusion(rankings: list[list[ScoredChunk]], k: int = 60) -> list[ScoredChunk]:
    """Combine multiple rankings of the same item type via Reciprocal Rank Fusion.

    RRF is robust to the very different score scales of BM25 vs. cosine
    similarity, since it only uses rank position: score(d) = sum(1 / (k + rank_i(d))).
    """
    fused_scores: dict[str, float] = {}
    chunk_by_id = {}
    methods_by_id: dict[str, set[str]] = {}

    for ranking in rankings:
        for rank, scored in enumerate(ranking):
            cid = scored.chunk.chunk_id
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunk_by_id[cid] = scored.chunk
            methods_by_id.setdefault(cid, set()).add(scored.method)

    fused = [
        ScoredChunk(chunk=chunk_by_id[cid], score=score, method="+".join(sorted(methods_by_id[cid])))
        for cid, score in fused_scores.items()
    ]
    fused.sort(key=lambda sc: sc.score, reverse=True)
    return fused
