"""FastAPI backend for the Cyber Threat Intelligence investigation dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cti_graphrag.corpus import Corpus, build_corpus
from cti_graphrag.llm import get_llm
from cti_graphrag.rag import SYSTEM_REGISTRY, BaseRAGSystem, build_all_systems

FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"
REPORTS_DIR = Path(__file__).resolve().parents[3] / "reports"

REPORT_FILES = {
    "eval": ("eval_results.json", "python -m cti_graphrag.evaluation.run_evaluation"),
    "ablation": ("ablation_results.json", "python -m cti_graphrag.evaluation.ablation"),
    "errors": ("error_analysis.json", "python -m cti_graphrag.evaluation.error_analysis"),
}

app = FastAPI(title="Explainable CTI Investigation API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_state: dict = {}


def get_corpus() -> Corpus:
    if "corpus" not in _state:
        _state["corpus"] = build_corpus()
    return _state["corpus"]


def get_systems() -> dict[str, BaseRAGSystem]:
    if "systems" not in _state:
        _state["systems"] = build_all_systems(get_corpus(), get_llm())
    return _state["systems"]


class QueryRequest(BaseModel):
    question: str
    system: str = "agentic_graph_rag"


class CompareRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/systems")
def list_systems() -> dict:
    return {"systems": list(SYSTEM_REGISTRY.keys())}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    systems = get_systems()
    system = systems.get(req.system)
    if system is None:
        raise HTTPException(status_code=400, detail=f"Unknown system '{req.system}'. Options: {list(systems)}")
    result = system.answer(req.question)
    return result.to_dict()


@app.post("/compare")
def compare(req: CompareRequest) -> dict:
    systems = get_systems()
    return {name: system.answer(req.question).to_dict() for name, system in systems.items()}


@app.get("/graph/entity")
def graph_entity(name: str, hops: int = 1) -> dict:
    corpus = get_corpus()
    matches = corpus.graph_store.find_nodes_by_name(name, limit=1)
    if not matches:
        raise HTTPException(status_code=404, detail=f"No graph entity found matching '{name}'")
    node = matches[0]

    from cti_graphrag.retrieval.graph_retriever import explore_subgraph

    paths = explore_subgraph(corpus.graph_store, [node], max_hops=hops)
    return {
        "entity": {"id": node.id, "name": node.name, "type": node.type.value},
        "paths": [p.to_dict() for p in paths if p.length > 0],
    }


@app.get("/reports/{name}")
def get_report(name: str) -> dict:
    entry = REPORT_FILES.get(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown report '{name}'. Options: {list(REPORT_FILES)}")
    filename, generate_cmd = entry
    path = REPORTS_DIR / filename
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report not generated yet. Run: {generate_cmd}",
        )
    return json.loads(path.read_text(encoding="utf-8"))


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
