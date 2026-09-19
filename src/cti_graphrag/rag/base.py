"""Common result type and base class shared by all four comparable RAG systems."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from cti_graphrag.graph.store import GraphPath
from cti_graphrag.llm import LLM
from cti_graphrag.rag.prompting import build_prompt
from cti_graphrag.retrieval.vector_store import ScoredChunk


@dataclass
class Citation:
    chunk_id: str
    source: str
    title: str
    text: str


@dataclass
class AnswerResult:
    """Uniform output of every RAG system: answer + evidence + explainable graph path."""

    system_name: str
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    graph_paths: list[GraphPath] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "system": self.system_name,
            "question": self.question,
            "answer": self.answer,
            "citations": [c.__dict__ for c in self.citations],
            "graph_paths": [p.to_dict() for p in self.graph_paths],
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "tool_calls": self.tool_calls,
            "latency_seconds": self.latency_seconds,
            "metadata": self.metadata,
        }


class BaseRAGSystem(ABC):
    system_name: str = "base"

    def __init__(self, llm: LLM):
        self._llm = llm

    def answer(self, question: str) -> AnswerResult:
        start = time.perf_counter()
        result = self._answer_impl(question)
        result.latency_seconds = time.perf_counter() - start
        return result

    @abstractmethod
    def _answer_impl(self, question: str) -> AnswerResult: ...

    def _generate(self, question: str, graph_paths: list[GraphPath], evidence: list[ScoredChunk]) -> str:
        system, user = build_prompt(question, graph_paths, evidence)
        return self._llm.generate(system, user)

    @staticmethod
    def _to_citations(evidence: list[ScoredChunk]) -> list[Citation]:
        return [
            Citation(
                chunk_id=sc.chunk.chunk_id,
                source=sc.chunk.metadata.get("source", "unknown"),
                title=sc.chunk.metadata.get("title", ""),
                text=sc.chunk.text,
            )
            for sc in evidence
        ]
