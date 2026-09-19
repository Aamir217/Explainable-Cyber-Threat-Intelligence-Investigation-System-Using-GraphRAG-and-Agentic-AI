"""System 2 -- Hybrid RAG: Vector + BM25 -> Reciprocal Rank Fusion -> Reranker -> LLM."""

from __future__ import annotations

from cti_graphrag.corpus import Corpus
from cti_graphrag.rag.base import AnswerResult, BaseRAGSystem
from cti_graphrag.retrieval.fusion import reciprocal_rank_fusion


class HybridRAG(BaseRAGSystem):
    system_name = "hybrid_rag"

    def __init__(self, corpus: Corpus, llm, top_k: int = 5, fusion_k: int = 10):
        super().__init__(llm)
        self._corpus = corpus
        self._top_k = top_k
        self._fusion_k = fusion_k

    def _answer_impl(self, question: str) -> AnswerResult:
        vector_hits = self._corpus.vector_store.search(question, top_k=self._fusion_k)
        bm25_hits = self._corpus.bm25_store.search(question, top_k=self._fusion_k)
        fused = reciprocal_rank_fusion([vector_hits, bm25_hits])
        evidence = self._corpus.reranker.rerank(question, fused, top_k=self._top_k)

        answer_text = self._generate(question, graph_paths=[], evidence=evidence)
        return AnswerResult(
            system_name=self.system_name,
            question=question,
            answer=answer_text,
            citations=self._to_citations(evidence),
            graph_paths=[],
            retrieved_chunk_ids=[sc.chunk.chunk_id for sc in evidence],
            tool_calls=["vector_search", "bm25_search", "fusion", "reranker"],
        )
