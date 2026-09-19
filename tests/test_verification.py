from cti_graphrag.graph.builder import build_knowledge_graph
from cti_graphrag.retrieval.graph_retriever import graph_search
from cti_graphrag.verification.claim_extraction import extract_claims
from cti_graphrag.verification.verifier import grounded_answer, verify_answer


def test_extract_claims_splits_sentences():
    claims = extract_claims("APT28 uses Zebrocy. Zebrocy exploits CVE-2020-0688!")
    assert claims == ["APT28 uses Zebrocy.", "Zebrocy exploits CVE-2020-0688!"]


def test_verify_answer_grounded_claim_is_supported():
    store = build_knowledge_graph()
    paths = graph_search(store, "Which malware does APT28 use?", max_hops=1)
    answer = "APT28 uses Zebrocy malware."
    report = verify_answer(answer, paths, chunks=[])
    assert report.faithfulness > 0.0
    assert any(c.supported for c in report.claims)


def test_verify_answer_flags_unsupported_claim():
    store = build_knowledge_graph()
    paths = graph_search(store, "Which malware does APT28 use?", max_hops=1)
    fabricated = "The moon is made of green cheese and unicorns run the internet."
    report = verify_answer(fabricated, paths, chunks=[])
    assert report.faithfulness < 1.0
    assert report.unsupported_claims


def test_grounded_answer_drops_unsupported_sentence():
    store = build_knowledge_graph()
    paths = graph_search(store, "Which malware does APT28 use?", max_hops=1)
    mixed = "APT28 uses Zebrocy malware. The moon is made of green cheese."
    report = verify_answer(mixed, paths, chunks=[])
    grounded = grounded_answer(mixed, report)
    assert "Zebrocy" in grounded
    assert "cheese" not in grounded
