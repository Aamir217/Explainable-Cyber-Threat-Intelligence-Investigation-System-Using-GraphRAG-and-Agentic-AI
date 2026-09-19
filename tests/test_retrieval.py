import pytest

from cti_graphrag.embeddings import get_embedder
from cti_graphrag.graph.builder import build_knowledge_graph
from cti_graphrag.ingestion.cti_reports import load_cti_reports
from cti_graphrag.retrieval.bm25_store import BM25Store
from cti_graphrag.retrieval.chunking import chunk_documents
from cti_graphrag.retrieval.fusion import reciprocal_rank_fusion
from cti_graphrag.retrieval.graph_retriever import graph_search, link_entities
from cti_graphrag.retrieval.reranker import Reranker
from cti_graphrag.retrieval.vector_store import VectorStore


@pytest.fixture(scope="module")
def chunks():
    return chunk_documents(load_cti_reports())


@pytest.fixture(scope="module")
def embedder():
    return get_embedder("hashing")


def test_chunking_preserves_metadata(chunks):
    assert len(chunks) > 0
    assert all(c.metadata.get("source") for c in chunks)
    assert all(c.chunk_id.startswith("RPT-") for c in chunks)


def test_vector_store_retrieves_relevant_chunk(chunks, embedder):
    vs = VectorStore(embedder)
    vs.index(chunks)
    results = vs.search("ProxyLogon Microsoft Exchange vulnerability", top_k=3)
    assert results
    assert any("CVE-2021-26855" in r.chunk.text for r in results)


def test_bm25_store_retrieves_relevant_chunk(chunks):
    bm = BM25Store()
    bm.index(chunks)
    results = bm.search("Zebrocy spearphishing", top_k=3)
    assert results
    assert any("Zebrocy" in r.chunk.text for r in results)


def test_fusion_combines_rankings(chunks, embedder):
    vs = VectorStore(embedder)
    vs.index(chunks)
    bm = BM25Store()
    bm.index(chunks)
    query = "APT28 Zebrocy exploit"
    fused = reciprocal_rank_fusion([vs.search(query, top_k=5), bm.search(query, top_k=5)])
    assert fused
    assert fused[0].score >= fused[-1].score


def test_reranker_orders_by_relevance(chunks, embedder):
    reranker = Reranker(embedder)
    vs = VectorStore(embedder)
    vs.index(chunks)
    candidates = vs.search("VMware Workspace ONE command injection", top_k=5)
    reranked = reranker.rerank("VMware Workspace ONE command injection", candidates, top_k=3)
    assert reranked
    assert any("CVE-2020-4006" in r.chunk.text for r in reranked)


def test_graph_search_multi_hop():
    store = build_knowledge_graph()
    linked = link_entities(store, "Which malware does APT28 use?")
    assert any(n.name == "APT28" for n in linked)

    paths = graph_search(store, "Which techniques are used by malware used by APT28?", max_hops=2)
    assert paths
    assert any("APT28" in p.to_text() for p in paths)
