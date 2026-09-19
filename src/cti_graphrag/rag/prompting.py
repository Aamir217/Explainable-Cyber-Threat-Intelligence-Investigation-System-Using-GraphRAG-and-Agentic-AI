"""Shared prompt construction so every RAG system feeds the LLM in the same, parseable layout."""

from __future__ import annotations

from cti_graphrag.graph.store import GraphPath
from cti_graphrag.retrieval.vector_store import ScoredChunk

SYSTEM_PROMPT = (
    "You are a cybersecurity threat intelligence analyst assistant. Answer the "
    "analyst's question strictly using the graph reasoning paths and document "
    "evidence provided. Every claim must be traceable to a graph path or a "
    "cited document chunk. If the evidence is insufficient, say so explicitly "
    "instead of guessing."
)


def build_prompt(question: str, graph_paths: list[GraphPath], evidence: list[ScoredChunk]) -> tuple[str, str]:
    graph_section = "\n".join(f"- {p.to_text()}" for p in graph_paths) or "- None."
    evidence_section = (
        "\n".join(f"- [{sc.chunk.chunk_id}] {sc.chunk.text}" for sc in evidence) if evidence else "- None."
    )

    user = (
        f"Question:\n{question}\n\n"
        f"Graph reasoning paths:\n{graph_section}\n\n"
        f"Supporting document evidence:\n{evidence_section}\n\n"
        "Instructions: Provide a concise, well-cited answer using only the information above."
    )
    return SYSTEM_PROMPT, user
