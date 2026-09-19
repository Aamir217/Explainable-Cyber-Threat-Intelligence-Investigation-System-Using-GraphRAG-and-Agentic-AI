from cti_graphrag.evaluation.metrics import (
    citation_correctness,
    exact_match,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    token_f1,
)


def test_exact_match():
    assert exact_match("APT28 uses Zebrocy", "apt28 uses zebrocy!") == 1.0
    assert exact_match("APT28 uses Zebrocy", "APT29 uses WellMess") == 0.0


def test_token_f1_partial_overlap():
    result = token_f1("APT28 uses Zebrocy and X-Agent", "APT28 uses Zebrocy")
    assert 0.0 < result["f1"] < 1.0
    assert result["recall"] == 1.0  # all gold tokens present


def test_precision_recall_at_k_dedupe_documents():
    # Metrics operate on doc-id-level items (chunk_id -> doc_id conversion
    # happens upstream in run_evaluation); RPT-001 appears twice here because
    # two chunks came from the same source document.
    retrieved = ["RPT-001", "RPT-001", "RPT-004"]
    relevant = {"RPT-001", "RPT-004"}
    assert precision_at_k(retrieved, relevant, k=5) == 1.0  # 2 unique docs, both relevant
    assert recall_at_k(retrieved, relevant, k=5) == 1.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank(["RPT-002", "RPT-001"], {"RPT-001"}) == 0.5
    assert mean_reciprocal_rank(["RPT-001"], {"RPT-001"}) == 1.0
    assert mean_reciprocal_rank(["RPT-003"], {"RPT-001"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    retrieved = ["RPT-001", "RPT-002"]
    relevant = {"RPT-001", "RPT-002"}
    assert ndcg_at_k(retrieved, relevant, k=2) == 1.0


def test_citation_correctness():
    assert citation_correctness(["RPT-001", "RPT-002"], ["RPT-001"]) == 0.5
    assert citation_correctness([], ["RPT-001"]) == 0.0
