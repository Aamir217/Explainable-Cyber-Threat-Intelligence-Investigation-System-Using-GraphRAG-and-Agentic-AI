"""System 1 -- Naive RAG: Vector Search -> LLM. The conventional baseline."""

from __future__ import annotations

from cti_graphrag.corpus import Corpus
from cti_graphrag.rag.base import AnswerResult, BaseRAGSystem


class NaiveRAG(BaseRAGSystem):
    system_name = "naive_rag"

    def __init__(self, corpus: Corpus, llm, top_k: int = 5):
        super().__init__(llm)
        self._corpus = corpus
        self._top_k = top_k

    def _answer_impl(self, question: str) -> AnswerResult:
        evidence = self._corpus.vector_store.search(question, top_k=self._top_k)
        answer_text = self._generate(question, graph_paths=[], evidence=evidence)
        return AnswerResult(
            system_name=self.system_name,
            question=question,
            answer=answer_text,
            citations=self._to_citations(evidence),
            graph_paths=[],
            retrieved_chunk_ids=[sc.chunk.chunk_id for sc in evidence],
            tool_calls=["vector_search"],
        )
