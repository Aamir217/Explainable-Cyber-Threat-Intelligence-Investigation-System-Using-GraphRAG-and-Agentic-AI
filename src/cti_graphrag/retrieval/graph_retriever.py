"""GraphRAG retrieval: entity linking + multi-hop subgraph exploration.

This is the component that lets the system answer questions vector/lexical
search cannot: it finds the entities mentioned in a query, walks the
knowledge graph outward from them, and returns the resulting reasoning
paths as both structured data (for the explainability panel) and text (to
feed the LLM as grounding context).
"""

from __future__ import annotations

import re

from cti_graphrag.embeddings import tokenize
from cti_graphrag.graph.schema import NodeType
from cti_graphrag.graph.store import GraphPath, GraphStore, PathStep


def link_entities(store: GraphStore, query: str, limit: int = 5):
    """Find graph nodes mentioned in free-text ``query`` (substring + fuzzy fallback)."""
    query_lower = query.lower()
    matches: list[tuple[int, object]] = []
    for node in store.all_nodes():
        for name in node.all_names():
            if len(name) < 3:
                continue
            if name.lower() in query_lower:
                matches.append((len(name), node))
                break

    matches.sort(key=lambda t: t[0], reverse=True)
    seen_ids: set[str] = set()
    linked = []
    for _, node in matches:
        if node.id not in seen_ids:
            seen_ids.add(node.id)
            linked.append(node)

    if not linked:
        linked = store.find_nodes_by_name(query, limit=limit)

    return linked[:limit]


def explore_subgraph(store: GraphStore, start_nodes, max_hops: int = 2, max_paths: int = 200) -> list[GraphPath]:
    """BFS outward (both directions) from each start node, up to ``max_hops``.

    Returns every intermediate path (not just leaves) so 1-hop, 2-hop, and
    3-hop reasoning chains are all available for downstream ranking.
    """
    all_paths: list[GraphPath] = [GraphPath(start=n) for n in start_nodes]
    frontier = list(all_paths)

    for _ in range(max_hops):
        next_frontier: list[GraphPath] = []
        for path in frontier:
            visited_ids = {n.id for n in path.nodes}
            for direction in ("out", "in"):
                for edge, node in store.neighbors(path.end.id, direction=direction):
                    if node.id in visited_ids:
                        continue
                    next_frontier.append(GraphPath(start=path.start, steps=[*path.steps, PathStep(edge, node)]))
        all_paths.extend(next_frontier)
        frontier = next_frontier
        if len(all_paths) >= max_paths:
            break

    return all_paths[:max_paths]


_TARGET_TYPE_KEYWORDS: dict[NodeType, list[str]] = {
    NodeType.CVE: ["vulnerabilit", "cve"],
    NodeType.MALWARE: ["malware"],
    NodeType.TECHNIQUE: ["technique"],
    NodeType.TACTIC: ["tactic"],
    NodeType.THREAT_ACTOR: ["threat actor", " actor", "group"],
    NodeType.PRODUCT: ["product"],
    NodeType.CAMPAIGN: ["campaign"],
    NodeType.SECTOR: ["sector"],
    NodeType.MITIGATION: ["mitigat"],
}


_WH_HEAD_RE = re.compile(
    r"^(?:which|what)\s+(.+?)\s+(?:are|is|do|does|use|uses|used|employ|employs|"
    r"exploit|exploits|exploited|affect|affects|target|targets)\b",
    re.IGNORECASE,
)


def _target_node_types(query: str) -> set[NodeType]:
    """Infer which node type(s) the question is actually asking for.

    A question can mention several entity types in its relationship chain
    (e.g. "which vulnerabilities are exploited by malware used by APT29")
    even though only one of them -- the head noun right after "which"/"what"
    -- is what should be *returned*. Falling back to "any keyword anywhere in
    the question" for questions that don't fit that pattern still handles the
    general case, just less precisely.
    """
    query_lower = query.lower()
    head_match = _WH_HEAD_RE.match(query_lower.strip())
    if head_match:
        head_phrase = head_match.group(1)
        matched = {t for t, kws in _TARGET_TYPE_KEYWORDS.items() if any(kw in head_phrase for kw in kws)}
        if matched:
            return matched
    return {t for t, keywords in _TARGET_TYPE_KEYWORDS.items() if any(kw in query_lower for kw in keywords)}


def rank_paths(paths: list[GraphPath], query: str, top_k: int = 10) -> list[GraphPath]:
    """Rank subgraph paths by how well they answer the query, not just by raw length.

    A question like "Which vulnerabilities are exploited by malware used by
    APT29?" names a target entity *type* (CVE) even though no specific CVE ID
    appears in the question text. Pure lexical overlap over node names cannot
    see that -- it would happily rank a long, tangential USES-chain above the
    short, directly-relevant "malware EXPLOITS CVE" path, since every hop
    shares generic tokens like "uses"/"malware"/the actor's name. Detecting
    the implied target type and rewarding paths that actually terminate on a
    node of that type (breaking ties by preferring the *shorter*, more direct
    explanation) fixes that without needing a real NLU model.
    """
    query_tokens = set(tokenize(query))
    target_types = _target_node_types(query)

    def score(path: GraphPath) -> float:
        path_tokens = set(tokenize(path.to_text()))
        node_overlap = len(query_tokens & path_tokens)
        end_type_match = 10.0 if (target_types and path.end.type in target_types) else 0.0
        length_penalty = 0.15 * path.length
        return end_type_match + node_overlap - length_penalty

    ranked = sorted(paths, key=score, reverse=True)
    return ranked[:top_k]


def graph_search(store: GraphStore, query: str, max_hops: int = 2, top_k: int = 10) -> list[GraphPath]:
    """End-to-end GraphRAG retrieval: link entities, expand, rank."""
    entities = link_entities(store, query)
    if not entities:
        return []
    paths = explore_subgraph(store, entities, max_hops=max_hops)
    # Drop zero-length (start-only) paths -- they carry no relationship evidence.
    paths = [p for p in paths if p.length > 0]
    return rank_paths(paths, query, top_k=top_k)
