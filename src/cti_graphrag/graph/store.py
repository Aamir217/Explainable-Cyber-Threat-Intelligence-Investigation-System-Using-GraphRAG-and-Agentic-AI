"""Graph storage backends.

``GraphStore`` is the abstraction the rest of the system (retrieval, agents,
API) programs against. Two implementations are provided:

* ``InMemoryGraphStore`` -- backed by ``networkx``, requires no external
  service, and is the default so the whole project runs out of the box.
* ``Neo4jGraphStore`` -- backed by a real Neo4j instance via the official
  Bolt driver, for production/enterprise deployments. It implements the same
  interface (including Cypher-based multi-hop traversal) so callers can swap
  backends via configuration alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from cti_graphrag.graph.schema import EdgeType, GraphEdge, GraphNode


@dataclass
class PathStep:
    edge: GraphEdge
    node: GraphNode


@dataclass
class GraphPath:
    """An explicit multi-hop reasoning path: start node, then (edge, node) hops."""

    start: GraphNode
    steps: list[PathStep] = field(default_factory=list)

    @property
    def end(self) -> GraphNode:
        return self.steps[-1].node if self.steps else self.start

    @property
    def nodes(self) -> list[GraphNode]:
        return [self.start, *[s.node for s in self.steps]]

    @property
    def length(self) -> int:
        return len(self.steps)

    def to_text(self) -> str:
        nodes = self.nodes
        parts = [nodes[0].name]
        for i, step in enumerate(self.steps):
            prev_node = nodes[i]
            forward = step.edge.source_id == prev_node.id
            arrow = f"--{step.edge.type.value}-->" if forward else f"<--{step.edge.type.value}--"
            parts.append(arrow)
            parts.append(step.node.name)
        return " ".join(parts)

    def to_dict(self) -> dict:
        nodes = self.nodes
        edges = []
        for i, step in enumerate(self.steps):
            edges.append(
                {
                    "type": step.edge.type.value,
                    "forward": step.edge.source_id == nodes[i].id,
                }
            )
        return {
            "nodes": [{"id": n.id, "type": n.type.value, "name": n.name} for n in nodes],
            "edges": edges,
            "text": self.to_text(),
        }


class GraphStore(ABC):
    @abstractmethod
    def add_nodes(self, nodes: Iterable[GraphNode]) -> None: ...

    @abstractmethod
    def add_edges(self, edges: Iterable[GraphEdge]) -> None: ...

    @abstractmethod
    def get_node(self, node_id: str) -> GraphNode | None: ...

    @abstractmethod
    def all_nodes(self) -> list[GraphNode]: ...

    @abstractmethod
    def all_edges(self) -> list[GraphEdge]: ...

    @abstractmethod
    def find_nodes_by_name(self, query: str, limit: int = 5) -> list[GraphNode]:
        """Fuzzy entity linking: find graph nodes whose name/aliases match ``query``."""
        ...

    @abstractmethod
    def neighbors(
        self,
        node_id: str,
        edge_types: Sequence[EdgeType] | None = None,
        direction: str = "out",
    ) -> list[tuple[GraphEdge, GraphNode]]: ...

    @abstractmethod
    def traverse(
        self,
        start_ids: Sequence[str],
        edge_type_sequence: Sequence[Sequence[EdgeType]],
        direction: str = "out",
    ) -> list[GraphPath]:
        """Follow a fixed sequence of edge-type hops from each start node.

        ``edge_type_sequence`` is a list of hops; each hop is the set of edge
        types allowed at that step. This is the primitive GraphRAG uses to
        answer templated multi-hop questions such as
        "actor --USES--> malware --USES--> technique".
        """
        ...

    @abstractmethod
    def shortest_path(self, start_id: str, end_id: str, max_hops: int = 6) -> GraphPath | None: ...


class InMemoryGraphStore(GraphStore):
    """Default backend: an in-process property graph built on networkx."""

    def __init__(self) -> None:
        import networkx as nx

        self._nx = nx
        self._graph = nx.MultiDiGraph()

    def add_nodes(self, nodes: Iterable[GraphNode]) -> None:
        for node in nodes:
            self._graph.add_node(node.id, data=node)

    def add_edges(self, edges: Iterable[GraphEdge]) -> None:
        for edge in edges:
            if edge.source_id not in self._graph or edge.target_id not in self._graph:
                # Skip dangling edges (e.g. a relationship referencing a node
                # from a source that has not been ingested yet).
                continue
            self._graph.add_edge(edge.source_id, edge.target_id, key=edge.type.value, data=edge)

    def get_node(self, node_id: str) -> GraphNode | None:
        node = self._graph.nodes.get(node_id)
        return node["data"] if node else None

    def all_nodes(self) -> list[GraphNode]:
        return [d["data"] for _, d in self._graph.nodes(data=True)]

    def all_edges(self) -> list[GraphEdge]:
        return [d["data"] for _, _, d in self._graph.edges(data=True)]

    def find_nodes_by_name(self, query: str, limit: int = 5) -> list[GraphNode]:
        query_norm = _normalize(query)
        scored: list[tuple[float, GraphNode]] = []
        for node in self.all_nodes():
            best = 0.0
            for name in node.all_names():
                name_norm = _normalize(name)
                score = _match_score(query_norm, name_norm)
                best = max(best, score)
            if best > 0:
                scored.append((best, node))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [n for _, n in scored[:limit]]

    def neighbors(
        self,
        node_id: str,
        edge_types: Sequence[EdgeType] | None = None,
        direction: str = "out",
    ) -> list[tuple[GraphEdge, GraphNode]]:
        results: list[tuple[GraphEdge, GraphNode]] = []
        allowed = {e.value for e in edge_types} if edge_types else None

        edge_iter = (
            self._graph.out_edges(node_id, data=True)
            if direction == "out"
            else self._graph.in_edges(node_id, data=True)
        )
        for u, v, d in edge_iter:
            edge: GraphEdge = d["data"]
            if allowed and edge.type.value not in allowed:
                continue
            other_id = v if direction == "out" else u
            other_node = self.get_node(other_id)
            if other_node is not None:
                results.append((edge, other_node))
        return results

    def traverse(
        self,
        start_ids: Sequence[str],
        edge_type_sequence: Sequence[Sequence[EdgeType]],
        direction: str = "out",
    ) -> list[GraphPath]:
        paths: list[GraphPath] = []

        def extend(current_path: GraphPath, remaining_hops: Sequence[Sequence[EdgeType]]) -> None:
            if not remaining_hops:
                paths.append(current_path)
                return
            hop_types, *rest = remaining_hops
            for edge, node in self.neighbors(current_path.end.id, edge_types=hop_types, direction=direction):
                new_path = GraphPath(start=current_path.start, steps=[*current_path.steps, PathStep(edge, node)])
                extend(new_path, rest)

        for sid in start_ids:
            start_node = self.get_node(sid)
            if start_node is None:
                continue
            extend(GraphPath(start=start_node), edge_type_sequence)

        return paths

    def shortest_path(self, start_id: str, end_id: str, max_hops: int = 6) -> GraphPath | None:
        try:
            node_path = self._nx.shortest_path(self._graph, start_id, end_id)
        except (self._nx.NetworkXNoPath, self._nx.NodeNotFound):
            return None
        if len(node_path) - 1 > max_hops:
            return None

        start_node = self.get_node(node_path[0])
        steps: list[PathStep] = []
        for u, v in zip(node_path, node_path[1:]):
            edge_data = self._graph.get_edge_data(u, v)
            first_edge: GraphEdge = next(iter(edge_data.values()))["data"]
            steps.append(PathStep(edge=first_edge, node=self.get_node(v)))
        return GraphPath(start=start_node, steps=steps)


def _normalize(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).strip()


def _match_score(query: str, candidate: str) -> float:
    if not query or not candidate:
        return 0.0
    if query == candidate:
        return 1.0
    if query in candidate or candidate in query:
        return 0.85
    query_tokens = set(query.split())
    cand_tokens = set(candidate.split())
    if not query_tokens or not cand_tokens:
        return 0.0
    overlap = len(query_tokens & cand_tokens)
    if overlap == 0:
        return 0.0
    return 0.5 * overlap / max(len(query_tokens), len(cand_tokens))


class Neo4jGraphStore(GraphStore):
    """Production backend using a real Neo4j database via the Bolt driver.

    Nodes are stored with a single label matching ``NodeType`` and an ``id``
    property; edges are stored as relationships typed by ``EdgeType``. This
    class requires the optional ``neo4j`` package and a reachable database
    (e.g. via ``docker compose up neo4j``, see ``docker-compose.yml``).
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
            raise ImportError("Install the 'neo4j' package to use Neo4jGraphStore") from exc

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def _run(self, query: str, **params):
        with self._driver.session(database=self._database) as session:
            return list(session.run(query, **params))

    def add_nodes(self, nodes: Iterable[GraphNode]) -> None:
        for node in nodes:
            self._run(
                f"MERGE (n:{node.type.value} {{id: $id}}) "
                "SET n.name = $name, n.aliases = $aliases, n += $properties",
                id=node.id,
                name=node.name,
                aliases=list(node.aliases),
                properties={k: v for k, v in node.properties.items() if isinstance(v, (str, int, float, bool))},
            )

    def add_edges(self, edges: Iterable[GraphEdge]) -> None:
        for edge in edges:
            self._run(
                "MATCH (a {id: $source_id}), (b {id: $target_id}) "
                f"MERGE (a)-[r:{edge.type.value}]->(b)",
                source_id=edge.source_id,
                target_id=edge.target_id,
            )

    def get_node(self, node_id: str) -> GraphNode | None:
        rows = self._run("MATCH (n {id: $id}) RETURN n, labels(n) AS labels", id=node_id)
        if not rows:
            return None
        return _neo4j_row_to_node(rows[0])

    def all_nodes(self) -> list[GraphNode]:
        rows = self._run("MATCH (n) RETURN n, labels(n) AS labels")
        return [_neo4j_row_to_node(r) for r in rows]

    def all_edges(self) -> list[GraphEdge]:
        rows = self._run("MATCH (a)-[r]->(b) RETURN a.id AS source_id, b.id AS target_id, type(r) AS rel_type")
        return [GraphEdge(r["source_id"], r["target_id"], EdgeType(r["rel_type"])) for r in rows]

    def find_nodes_by_name(self, query: str, limit: int = 5) -> list[GraphNode]:
        rows = self._run(
            "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($q) "
            "OR any(a IN coalesce(n.aliases, []) WHERE toLower(a) CONTAINS toLower($q)) "
            "RETURN n, labels(n) AS labels LIMIT $limit",
            q=query,
            limit=limit,
        )
        return [_neo4j_row_to_node(r) for r in rows]

    def neighbors(
        self,
        node_id: str,
        edge_types: Sequence[EdgeType] | None = None,
        direction: str = "out",
    ) -> list[tuple[GraphEdge, GraphNode]]:
        rel_filter = "|".join(e.value for e in edge_types) if edge_types else ""
        rel_pattern = f"[r:{rel_filter}]" if rel_filter else "[r]"
        pattern = f"(a {{id: $id}})-{rel_pattern}->(b)" if direction == "out" else f"(a {{id: $id}})<-{rel_pattern}-(b)"
        rows = self._run(f"MATCH {pattern} RETURN r, type(r) AS rel_type, b, labels(b) AS labels", id=node_id)
        results = []
        for r in rows:
            edge = GraphEdge(node_id, r["b"]["id"], EdgeType(r["rel_type"])) if direction == "out" else GraphEdge(
                r["b"]["id"], node_id, EdgeType(r["rel_type"])
            )
            results.append((edge, _neo4j_row_to_node({"n": r["b"], "labels": r["labels"]})))
        return results

    def traverse(
        self,
        start_ids: Sequence[str],
        edge_type_sequence: Sequence[Sequence[EdgeType]],
        direction: str = "out",
    ) -> list[GraphPath]:
        # Delegate to generic BFS using neighbors() so Cypher stays simple and
        # the traversal semantics match InMemoryGraphStore exactly.
        paths: list[GraphPath] = []

        def extend(current_path: GraphPath, remaining_hops: Sequence[Sequence[EdgeType]]) -> None:
            if not remaining_hops:
                paths.append(current_path)
                return
            hop_types, *rest = remaining_hops
            for edge, node in self.neighbors(current_path.end.id, edge_types=hop_types, direction=direction):
                extend(GraphPath(start=current_path.start, steps=[*current_path.steps, PathStep(edge, node)]), rest)

        for sid in start_ids:
            start_node = self.get_node(sid)
            if start_node is not None:
                extend(GraphPath(start=start_node), edge_type_sequence)
        return paths

    def shortest_path(self, start_id: str, end_id: str, max_hops: int = 6) -> GraphPath | None:
        rows = self._run(
            "MATCH p = shortestPath((a {id: $start_id})-[*..%d]->(b {id: $end_id})) "
            "RETURN [n IN nodes(p) | n] AS nodes, [n IN nodes(p) | labels(n)] AS labels, "
            "[r IN relationships(p) | type(r)] AS rel_types" % max_hops,
            start_id=start_id,
            end_id=end_id,
        )
        if not rows:
            return None
        row = rows[0]
        node_dicts, label_lists, rel_types = row["nodes"], row["labels"], row["rel_types"]
        nodes = [_neo4j_row_to_node({"n": n, "labels": labels}) for n, labels in zip(node_dicts, label_lists)]
        steps = [
            PathStep(edge=GraphEdge(nodes[i].id, nodes[i + 1].id, EdgeType(rel_types[i])), node=nodes[i + 1])
            for i in range(len(rel_types))
        ]
        return GraphPath(start=nodes[0], steps=steps)


def _neo4j_row_to_node(row) -> GraphNode:
    from cti_graphrag.graph.schema import NodeType

    props = dict(row["n"])
    labels = row["labels"]
    node_type = NodeType(labels[0]) if labels else NodeType.THREAT_ACTOR
    return GraphNode(
        id=props.pop("id"),
        type=node_type,
        name=props.pop("name", ""),
        aliases=tuple(props.pop("aliases", []) or []),
        properties=props,
    )
