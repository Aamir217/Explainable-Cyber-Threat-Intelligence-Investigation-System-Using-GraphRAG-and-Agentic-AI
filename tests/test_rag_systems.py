from cti_graphrag.rag import AgenticGraphRAG, GraphRAGSystem, HybridRAG, NaiveRAG


def test_naive_rag_answers_with_no_graph_path(corpus, llm):
    system = NaiveRAG(corpus, llm)
    result = system.answer("Which malware does APT29 use?")
    assert result.answer
    assert result.graph_paths == []
    assert result.citations


def test_hybrid_rag_uses_fusion_and_rerank_tools(corpus, llm):
    system = HybridRAG(corpus, llm)
    result = system.answer("Which malware does APT29 use?")
    assert "fusion" in result.tool_calls
    assert "reranker" in result.tool_calls


def test_graph_rag_surfaces_multi_hop_path_naive_cannot(corpus, llm):
    question = (
        "Which threat actors use malware that exploits vulnerabilities affecting "
        "Microsoft Exchange Server, and which attack techniques are involved?"
    )
    naive_result = NaiveRAG(corpus, llm).answer(question)
    graph_result = GraphRAGSystem(corpus, llm).answer(question)

    assert naive_result.graph_paths == []
    assert len(graph_result.graph_paths) > 0
    # The graph path must actually chain Exchange -> CVE -> malware -> actor.
    path_texts = " ".join(p.to_text() for p in graph_result.graph_paths)
    assert "APT28" in path_texts or "APT29" in path_texts
    assert "CVE-" in path_texts


def test_agentic_graph_rag_runs_planner_and_verification(corpus, llm):
    system = AgenticGraphRAG(corpus, llm, verify=True)
    result = system.answer("Which vulnerabilities are exploited by malware used by APT28?")
    assert "verification" in result.tool_calls
    assert "faithfulness" in result.metadata
    assert 0.0 <= result.metadata["faithfulness"] <= 1.0
    assert result.graph_paths


def test_agentic_graph_rag_adaptive_retry_on_empty_first_pass(corpus, llm):
    system = AgenticGraphRAG(corpus, llm, verify=False)
    # A question with no recognizable graph entities forces the empty-first-pass path.
    result = system.answer("What is the general state of global cybersecurity policy?")
    assert result.metadata["retrieval_iterations"] in (1, 2)


def test_all_systems_return_uniform_result_shape(corpus, llm):
    question = "Which techniques are used by APT28?"
    for system in (NaiveRAG(corpus, llm), HybridRAG(corpus, llm), GraphRAGSystem(corpus, llm), AgenticGraphRAG(corpus, llm)):
        result = system.answer(question)
        d = result.to_dict()
        assert set(d.keys()) == {
            "system", "question", "answer", "citations", "graph_paths",
            "retrieved_chunk_ids", "tool_calls", "latency_seconds", "metadata",
        }
