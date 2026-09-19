from .agentic_graph_rag import AgenticGraphRAG
from .base import AnswerResult, BaseRAGSystem, Citation
from .graph_rag import GraphRAGSystem
from .hybrid_rag import HybridRAG
from .naive_rag import NaiveRAG

SYSTEM_REGISTRY = {
    NaiveRAG.system_name: NaiveRAG,
    HybridRAG.system_name: HybridRAG,
    GraphRAGSystem.system_name: GraphRAGSystem,
    AgenticGraphRAG.system_name: AgenticGraphRAG,
}


def build_all_systems(corpus, llm) -> dict[str, BaseRAGSystem]:
    return {name: cls(corpus, llm) for name, cls in SYSTEM_REGISTRY.items()}


__all__ = [
    "SYSTEM_REGISTRY",
    "AgenticGraphRAG",
    "AnswerResult",
    "BaseRAGSystem",
    "Citation",
    "GraphRAGSystem",
    "HybridRAG",
    "NaiveRAG",
    "build_all_systems",
]
