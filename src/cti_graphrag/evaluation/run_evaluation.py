"""Run all four RAG systems over the evaluation set and compare them quantitatively.

Usage:
    python -m cti_graphrag.evaluation.run_evaluation [--output reports/eval_results.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from cti_graphrag.corpus import build_corpus
from cti_graphrag.evaluation.dataset import EvalQuestion, load_eval_questions
from cti_graphrag.evaluation.metrics import (
    citation_correctness,
    entity_coverage,
    exact_match,
    faithfulness,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    semantic_similarity,
    token_f1,
)
from cti_graphrag.llm import get_llm
from cti_graphrag.rag import AnswerResult, build_all_systems

DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "reports" / "eval_results.json"


def _doc_id_of(chunk_id: str) -> str:
    return chunk_id.split("::")[0]


def score_result(question: EvalQuestion, result: AnswerResult, embedder) -> dict:
    retrieved_doc_ids = [_doc_id_of(cid) for cid in result.retrieved_chunk_ids]
    relevant_docs = set(question.supporting_documents)

    f1 = token_f1(result.answer, question.ground_truth_answer)
    evidence_texts = [c.text for c in result.citations] + [p.to_text() for p in result.graph_paths]

    return {
        "question_id": question.id,
        "category": question.category,
        "exact_match": exact_match(result.answer, question.ground_truth_answer),
        "f1": f1["f1"],
        "semantic_similarity": semantic_similarity(result.answer, question.ground_truth_answer, embedder),
        "entity_coverage": entity_coverage(result.answer, question.relevant_entities),
        "precision_at_5": precision_at_k(retrieved_doc_ids, relevant_docs, 5),
        "recall_at_5": recall_at_k(retrieved_doc_ids, relevant_docs, 5),
        "mrr": mean_reciprocal_rank(retrieved_doc_ids, relevant_docs),
        "ndcg_at_5": ndcg_at_k(retrieved_doc_ids, relevant_docs, 5),
        "citation_correctness": citation_correctness(retrieved_doc_ids, question.supporting_documents),
        "faithfulness": faithfulness(result.answer, evidence_texts),
        "num_graph_paths": len(result.graph_paths),
        "latency_seconds": result.latency_seconds,
        "tool_calls": result.tool_calls,
    }


def run_evaluation(output_path: Path | None = None) -> dict:
    corpus = build_corpus()
    llm = get_llm()
    systems = build_all_systems(corpus, llm)
    questions = load_eval_questions()

    per_row: list[dict] = []
    for question in questions:
        for system_name, system in systems.items():
            result = system.answer(question.question)
            row = score_result(question, result, corpus.embedder)
            row["system"] = system_name
            per_row.append(row)

    summary = summarize(per_row)

    report = {"rows": per_row, "summary": summary}
    output_path = output_path or DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    return report


def summarize(rows: list[dict]) -> dict:
    numeric_fields = [
        "exact_match", "f1", "semantic_similarity", "entity_coverage",
        "precision_at_5", "recall_at_5", "mrr", "ndcg_at_5",
        "citation_correctness", "faithfulness", "latency_seconds", "num_graph_paths",
    ]

    by_system: dict[str, list[dict]] = defaultdict(list)
    by_system_category: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(row)
        by_system_category[(row["system"], row["category"])].append(row)

    def avg(rows_subset: list[dict]) -> dict:
        return {f: round(statistics.mean(r[f] for r in rows_subset), 4) for f in numeric_fields}

    return {
        "overall_by_system": {system: avg(rs) for system, rs in by_system.items()},
        "by_system_and_category": {
            f"{system}/{category}": avg(rs) for (system, category), rs in by_system_category.items()
        },
    }


def print_summary(report: dict) -> None:
    print("\n=== Overall averages by system ===")
    header_printed = False
    for system, metrics in report["summary"]["overall_by_system"].items():
        if not header_printed:
            print(f"{'system':22s} " + " ".join(f"{k:>18s}" for k in metrics))
            header_printed = True
        print(f"{system:22s} " + " ".join(f"{v:>18.4f}" for v in metrics.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = run_evaluation(args.output)
    print_summary(report)
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
