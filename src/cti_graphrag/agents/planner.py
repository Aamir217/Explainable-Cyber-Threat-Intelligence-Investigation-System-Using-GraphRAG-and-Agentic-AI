"""AI Planner: decides which retrieval tools a query needs and how deep to traverse.

This is a transparent, rule-based planner rather than an LLM-driven ReAct
loop, which keeps agentic behavior deterministic and reproducible for the
evaluation/ablation study (Section 12 of the project spec explicitly
compares "Agentic GraphRAG" against fixed pipelines -- that comparison is
only meaningful if the agent's decisions are inspectable and repeatable).
It still performs genuine adaptive planning: hop depth scales with query
complexity, the graph tool is skipped when no entities are recognized, and
``AgenticGraphRAG`` uses this plan to run a second, deeper retrieval
iteration when the first pass comes back empty. Swapping in an LLM-based
planner is a drop-in change: implement the same ``plan(question) -> Plan``
signature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cti_graphrag.retrieval.graph_retriever import link_entities

_RELATIONSHIP_KEYWORDS = [
    "use", "uses", "used", "exploit", "exploits", "exploited",
    "associate", "associated", "connect", "connected",
    "target", "targets", "employ", "employs", "affect", "affects",
]
_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


@dataclass
class Plan:
    tools: list[str] = field(default_factory=list)
    graph_hops: int = 1
    cve_ids: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)


def plan(question: str, corpus) -> Plan:
    q_lower = question.lower()
    rationale: list[str] = []

    hop_signal = sum(1 for kw in _RELATIONSHIP_KEYWORDS if kw in q_lower)
    graph_hops = min(4, max(1, hop_signal))
    rationale.append(f"detected {hop_signal} relationship keyword(s) -> graph_hops={graph_hops}")

    cve_ids = _CVE_RE.findall(question)
    if cve_ids:
        rationale.append(f"detected explicit CVE id(s): {cve_ids}")

    linked = link_entities(corpus.graph_store, question)
    tools = ["vector_search", "bm25_search"]
    if linked:
        tools.insert(0, "graph_search")
        rationale.append(f"linked {len(linked)} graph entit(y/ies): {[n.name for n in linked]}")
    else:
        rationale.append("no graph entities recognized in query -> skipping graph_search")

    if cve_ids:
        tools.append("cve_lookup")

    return Plan(tools=tools, graph_hops=graph_hops, cve_ids=cve_ids, rationale=rationale)
