"""Tools the agentic planner can invoke, wrapping the underlying retrieval stack."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from cti_graphrag.corpus import Corpus
from cti_graphrag.graph.schema import NodeType
from cti_graphrag.graph.store import GraphPath
from cti_graphrag.retrieval.fusion import reciprocal_rank_fusion
from cti_graphrag.retrieval.graph_retriever import graph_search
from cti_graphrag.retrieval.vector_store import ScoredChunk


@dataclass
class ToolResult:
    tool_name: str
    graph_paths: list[GraphPath] = field(default_factory=list)
    chunks: list[ScoredChunk] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Tool(ABC):
    name: str

    @abstractmethod
    def run(self, question: str, corpus: Corpus, **kwargs) -> ToolResult: ...


class GraphSearchTool(Tool):
    name = "graph_search"

    def run(self, question: str, corpus: Corpus, max_hops: int = 2, top_k: int = 10, **kwargs) -> ToolResult:
        paths = graph_search(corpus.graph_store, question, max_hops=max_hops, top_k=top_k)
        return ToolResult(tool_name=self.name, graph_paths=paths)


class VectorSearchTool(Tool):
    name = "vector_search"

    def run(self, question: str, corpus: Corpus, top_k: int = 8, **kwargs) -> ToolResult:
        chunks = corpus.vector_store.search(question, top_k=top_k)
        return ToolResult(tool_name=self.name, chunks=chunks)


class BM25SearchTool(Tool):
    name = "bm25_search"

    def run(self, question: str, corpus: Corpus, top_k: int = 8, **kwargs) -> ToolResult:
        chunks = corpus.bm25_store.search(question, top_k=top_k)
        return ToolResult(tool_name=self.name, chunks=chunks)


class CVELookupTool(Tool):
    """Direct vulnerability lookup: resolves a CVE ID to its graph node and affected products."""

    name = "cve_lookup"

    def run(self, question: str, corpus: Corpus, cve_ids: list[str] | None = None, **kwargs) -> ToolResult:
        from cti_graphrag.ingestion.nvd_cve import cve_node_id

        notes = []
        paths = []
        for cve_id in cve_ids or []:
            node = corpus.graph_store.get_node(cve_node_id(cve_id))
            if node is None:
                notes.append(f"{cve_id} not found in the knowledge graph.")
                continue
            severity = node.properties.get("cvss_v3_severity", "unknown")
            score = node.properties.get("cvss_v3_score", "unknown")
            notes.append(f"{cve_id}: severity={severity}, CVSSv3={score}. {node.properties.get('description', '')}")
            affected = corpus.graph_store.neighbors(node.id, direction="out")
            for edge, product in affected:
                if product.type is NodeType.PRODUCT:
                    from cti_graphrag.graph.store import GraphPath, PathStep

                    paths.append(GraphPath(start=node, steps=[PathStep(edge=edge, node=product)]))
        return ToolResult(tool_name=self.name, graph_paths=paths, notes=notes)


def fuse_chunks(*tool_results: ToolResult, k: int = 60) -> list[ScoredChunk]:
    rankings = [tr.chunks for tr in tool_results if tr.chunks]
    return reciprocal_rank_fusion(rankings, k=k) if rankings else []


DEFAULT_TOOLS: dict[str, Tool] = {
    tool.name: tool for tool in (GraphSearchTool(), VectorSearchTool(), BM25SearchTool(), CVELookupTool())
}
