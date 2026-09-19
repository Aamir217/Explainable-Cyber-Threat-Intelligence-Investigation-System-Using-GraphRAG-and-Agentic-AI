"""Ablation study (Section 12): incrementally add components and measure the effect.

Naive RAG -> Hybrid RAG -> GraphRAG -> GraphRAG+Reranking -> GraphRAG+Hybrid
Retrieval -> Agentic GraphRAG -> Agentic GraphRAG+Verification

Usage:
    python -m cti_graphrag.evaluation.ablation [--output reports/ablation_results.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cti_graphrag.corpus import build_corpus
from cti_graphrag.evaluation.dataset import load_eval_questions
from cti_graphrag.evaluation.run_evaluation import score_result, summarize
from cti_graphrag.llm import get_llm
from cti_graphrag.rag.agentic_graph_rag import AgenticGraphRAG
from cti_graphrag.rag.graph_rag import GraphRAGSystem
from cti_graphrag.rag.hybrid_rag import HybridRAG
from cti_graphrag.rag.naive_rag import NaiveRAG

DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "reports" / "ablation_results.json"

# Ordered so the report tells a story: each row adds exactly one capability
# relative to the row above it.
ABLATION_ORDER = [
    "naive_rag",
    "hybrid_rag",
    "graph_rag",
    "graph_rag_reranked",
    "graph_rag_hybrid_retrieval",
    "agentic_graph_rag_no_verification",
    "agentic_graph_rag_verified",
]


def build_ablation_systems(corpus, llm) -> dict:
    return {
        "naive_rag": NaiveRAG(corpus, llm),
        "hybrid_rag": HybridRAG(corpus, llm),
        "graph_rag": GraphRAGSystem(corpus, llm),
        "graph_rag_reranked": GraphRAGSystem(corpus, llm, rerank=True, name="graph_rag_reranked"),
        "graph_rag_hybrid_retrieval": GraphRAGSystem(
            corpus, llm, use_hybrid_docs=True, rerank=True, name="graph_rag_hybrid_retrieval"
        ),
        "agentic_graph_rag_no_verification": AgenticGraphRAG(
            corpus, llm, verify=False, name="agentic_graph_rag_no_verification"
        ),
        "agentic_graph_rag_verified": AgenticGraphRAG(corpus, llm, verify=True, name="agentic_graph_rag_verified"),
    }


def run_ablation(output_path: Path | None = None) -> dict:
    corpus = build_corpus()
    llm = get_llm()
    systems = build_ablation_systems(corpus, llm)
    questions = load_eval_questions()

    rows = []
    for question in questions:
        for name in ABLATION_ORDER:
            result = systems[name].answer(question.question)
            row = score_result(question, result, corpus.embedder)
            row["system"] = name
            rows.append(row)

    summary = summarize(rows)
    ordered_summary = {name: summary["overall_by_system"][name] for name in ABLATION_ORDER}

    report = {"rows": rows, "summary": {"overall_by_system": ordered_summary}}
    output_path = output_path or DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    return report


def print_ablation(report: dict) -> None:
    print("\n=== Ablation study: incremental component effect ===")
    metrics = ["f1", "semantic_similarity", "entity_coverage", "recall_at_5", "faithfulness", "latency_seconds"]
    print(f"{'configuration':38s} " + " ".join(f"{m:>18s}" for m in metrics))
    for name in ABLATION_ORDER:
        vals = report["summary"]["overall_by_system"][name]
        print(f"{name:38s} " + " ".join(f"{vals[m]:>18.4f}" for m in metrics))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = run_ablation(args.output)
    print_ablation(report)
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
