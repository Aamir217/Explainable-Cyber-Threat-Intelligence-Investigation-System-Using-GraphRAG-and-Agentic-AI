"""Load the evaluation question set."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_EVAL_PATH = Path(__file__).resolve().parents[3] / "data" / "eval" / "eval_questions.json"


@dataclass
class EvalQuestion:
    id: str
    category: str
    question: str
    ground_truth_answer: str
    relevant_entities: list[str] = field(default_factory=list)
    required_relationships: list[str] = field(default_factory=list)
    supporting_documents: list[str] = field(default_factory=list)
    expected_reasoning_path: list[str] = field(default_factory=list)


def load_eval_questions(path: Path | None = None) -> list[EvalQuestion]:
    path = path or DEFAULT_EVAL_PATH
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [EvalQuestion(**q) for q in data["questions"]]
