"""Categorize incorrect answers into the failure taxonomy from Section 13 of the spec.

This is a heuristic classifier -- it inspects what each system actually
retrieved/generated relative to the gold annotation and assigns zero or more
applicable categories. It is meant to explain *why* a system scored poorly on
a question, not to be a perfectly calibrated diagnostic tool.

Usage:
    python -m cti_graphrag.evaluation.error_analysis [--output reports/error_analysis.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cti_graphrag.corpus import Corpus, build_corpus
from cti_graphrag.evaluation.dataset import EvalQuestion, load_eval_questions
from cti_graphrag.evaluation.metrics import citation_correctness, entity_coverage, faithfulness, token_f1
from cti_graphrag.llm import get_llm
from cti_graphrag.rag import AnswerResult, build_all_systems
from cti_graphrag.retrieval.graph_retriever import link_entities

DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "reports" / "error_analysis.json"

INCORRECT_F1_THRESHOLD = 0.3


def categorize(question: EvalQuestion, result: AnswerResult, corpus: Corpus) -> list[str]:
    categories: list[str] = []

    retrieved_doc_ids = [cid.split("::")[0] for cid in result.retrieved_chunk_ids]
    linked = link_entities(corpus.graph_store, question.question)
    linked_names = {n.name for n in linked}
    expected_start_entities = set(question.expected_reasoning_path[:1])

    if not result.graph_paths and not result.citations:
        categories.append("insufficient_evidence")

    if question.expected_reasoning_path and not (linked_names & expected_start_entities):
        categories.append("entity_recognition_failure")

    if linked and not result.graph_paths and len(question.required_relationships) > 0:
        categories.append("missing_graph_relationship_or_incorrect_traversal")

    if question.supporting_documents and retrieved_doc_ids:
        if not (set(retrieved_doc_ids) & set(question.supporting_documents)):
            categories.append("document_retrieval_failure")

    if result.citations and citation_correctness(retrieved_doc_ids, question.supporting_documents) < 0.5:
        categories.append("incorrect_citation")

    evidence_texts = [c.text for c in result.citations] + [p.to_text() for p in result.graph_paths]
    if faithfulness(result.answer, evidence_texts) < 0.7:
        categories.append("hallucination")

    coverage = entity_coverage(result.answer, question.relevant_entities)
    if coverage < 0.5 and (result.graph_paths or result.citations):
        categories.append("reasoning_failure")

    return categories or ["none"]


def run_error_analysis(output_path: Path | None = None) -> dict:
    corpus = build_corpus()
    llm = get_llm()
    systems = build_all_systems(corpus, llm)
    questions = load_eval_questions()

    records = []
    counts: dict[str, Counter] = {name: Counter() for name in systems}

    for question in questions:
        for system_name, system in systems.items():
            result = system.answer(question.question)
            f1 = token_f1(result.answer, question.ground_truth_answer)["f1"]
            if f1 >= INCORRECT_F1_THRESHOLD:
                continue  # only analyze answers judged incorrect
            cats = categorize(question, result, corpus)
            counts[system_name].update(cats)
            records.append(
                {
                    "question_id": question.id,
                    "category": question.category,
                    "system": system_name,
                    "f1": f1,
                    "error_categories": cats,
                    "answer": result.answer,
                }
            )

    report = {
        "incorrect_answers": records,
        "error_category_counts_by_system": {k: dict(v) for k, v in counts.items()},
    }
    output_path = output_path or DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    return report


def print_error_analysis(report: dict) -> None:
    print("\n=== Error category counts by system (answers with F1 < %.2f) ===" % INCORRECT_F1_THRESHOLD)
    for system, cats in report["error_category_counts_by_system"].items():
        print(f"{system}: {dict(cats) if cats else 'no incorrect answers'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = run_error_analysis(args.output)
    print_error_analysis(report)
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
