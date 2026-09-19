"""System 3 -- GraphRAG: Graph Retrieval + Document Retrieval -> LLM.

Adds multi-hop knowledge-graph traversal alongside document retrieval, but
(unlike Agentic GraphRAG) always runs both retrievers in a fixed pipeline --
there is no planner deciding when graph traversal is actually needed, and no
verification pass.
"""

from __future__ import annotations

from cti_graphrag.config import settings
from cti_graphrag.corpus import Corpus
from cti_graphrag.rag.base import AnswerResult, BaseRAGSystem
from cti_graphrag.retrieval.fusion import reciprocal_rank_fusion
from cti_graphrag.retrieval.graph_retriever import graph_search


class GraphRAGSystem(BaseRAGSystem):
    """GraphRAG with two optional knobs (used by the ablation study, Section 12):

    * ``use_hybrid_docs`` -- fuse vector + BM25 document retrieval instead of vector-only.
    * ``rerank`` -- apply the cross-encoder-style reranker to the document evidence.
    """

    system_name = "graph_rag"

    def __init__(
        self,
        corpus: Corpus,
        llm,
        top_k: int = 5,
        max_hops: int | None = None,
        use_hybrid_docs: bool = False,
        rerank: bool = False,
        name: str | None = None,
    ):
        super().__init__(llm)
        self._corpus = corpus
        self._top_k = top_k
        self._max_hops = max_hops or settings.max_graph_hops
        self._use_hybrid_docs = use_hybrid_docs
        self._rerank = rerank
        if name:
            self.system_name = name

    def _retrieve_documents(self, question: str):
        fusion_k = self._top_k * 2
        if self._use_hybrid_docs:
            vector_hits = self._corpus.vector_store.search(question, top_k=fusion_k)
            bm25_hits = self._corpus.bm25_store.search(question, top_k=fusion_k)
            candidates = reciprocal_rank_fusion([vector_hits, bm25_hits])
            tool_calls = ["vector_search", "bm25_search", "fusion"]
        else:
            candidates = self._corpus.vector_store.search(question, top_k=fusion_k if self._rerank else self._top_k)
            tool_calls = ["vector_search"]

        if self._rerank:
            evidence = self._corpus.reranker.rerank(question, candidates, top_k=self._top_k)
            tool_calls.append("reranker")
        else:
            evidence = candidates[: self._top_k]
        return evidence, tool_calls

    def _answer_impl(self, question: str) -> AnswerResult:
        graph_paths = graph_search(self._corpus.graph_store, question, max_hops=self._max_hops, top_k=self._top_k)
        evidence, doc_tool_calls = self._retrieve_documents(question)

        answer_text = self._generate(question, graph_paths=graph_paths, evidence=evidence)
        return AnswerResult(
            system_name=self.system_name,
            question=question,
            answer=answer_text,
            citations=self._to_citations(evidence),
            graph_paths=graph_paths,
            retrieved_chunk_ids=[sc.chunk.chunk_id for sc in evidence],
            tool_calls=["graph_search", *doc_tool_calls],
        )
