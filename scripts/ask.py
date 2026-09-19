#!/usr/bin/env python3
"""CLI: ask a single question of one (or all) of the four RAG systems.

Examples:
    python scripts/ask.py "Which malware is used by APT28?"
    python scripts/ask.py "Which malware is used by APT28?" --system graph_rag
    python scripts/ask.py "Which malware is used by APT28?" --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cti_graphrag.corpus import build_corpus  # noqa: E402
from cti_graphrag.llm import get_llm  # noqa: E402
from cti_graphrag.rag import SYSTEM_REGISTRY, build_all_systems  # noqa: E402


def print_result(result) -> None:
    print(f"\n=== {result.system_name} ({result.latency_seconds * 1000:.1f} ms) ===")
    print(f"Tools used: {', '.join(result.tool_calls)}")
    print(f"\nAnswer:\n{result.answer}\n")
    if result.graph_paths:
        print("Graph reasoning paths:")
        for p in result.graph_paths:
            print(f"  - {p.to_text()}")
    if result.citations:
        print("\nEvidence:")
        for c in result.citations:
            print(f"  - [{c.chunk_id}] ({c.source}) {c.text[:120]}...")
    if result.metadata:
        print(f"\nMetadata: {result.metadata}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--system", default="agentic_graph_rag", choices=list(SYSTEM_REGISTRY))
    parser.add_argument("--all", action="store_true", help="Run all four systems for comparison")
    args = parser.parse_args()

    corpus = build_corpus()
    llm = get_llm()
    systems = build_all_systems(corpus, llm)

    if args.all:
        for system in systems.values():
            print_result(system.answer(args.question))
    else:
        print_result(systems[args.system].answer(args.question))


if __name__ == "__main__":
    main()
