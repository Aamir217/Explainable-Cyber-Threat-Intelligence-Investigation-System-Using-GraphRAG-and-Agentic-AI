"""Builds and bundles the shared knowledge graph + document indices used by every RAG system."""

from __future__ import annotations

from dataclasses import dataclass

from cti_graphrag.config import settings
from cti_graphrag.embeddings import Embedder, get_embedder
from cti_graphrag.graph.builder import build_knowledge_graph
from cti_graphrag.graph.store import GraphStore, Neo4jGraphStore
from cti_graphrag.ingestion.cti_reports import load_cti_reports
from cti_graphrag.retrieval.bm25_store import BM25Store
from cti_graphrag.retrieval.chunking import Chunk, chunk_documents
from cti_graphrag.retrieval.reranker import Reranker
from cti_graphrag.retrieval.vector_store import VectorStore


@dataclass
class Corpus:
    graph_store: GraphStore
    vector_store: VectorStore
    bm25_store: BM25Store
    embedder: Embedder
    reranker: Reranker
    chunks: list[Chunk]


def _make_graph_store() -> GraphStore:
    if settings.graph_backend == "neo4j":
        return Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    return None  # let build_knowledge_graph create the default InMemoryGraphStore


def build_corpus(use_live_sources: bool = False, embedding_backend: str | None = None) -> Corpus:
    graph_store = build_knowledge_graph(store=_make_graph_store(), use_live_sources=use_live_sources)

    docs = load_cti_reports()
    chunks = chunk_documents(docs)

    embedder = get_embedder(embedding_backend or settings.embedding_backend)
    vector_store = VectorStore(embedder)
    vector_store.index(chunks)

    bm25_store = BM25Store()
    bm25_store.index(chunks)

    reranker = Reranker(embedder)

    return Corpus(
        graph_store=graph_store,
        vector_store=vector_store,
        bm25_store=bm25_store,
        embedder=embedder,
        reranker=reranker,
        chunks=chunks,
    )
