"""Ingest MITRE ATT&CK data (simplified STIX bundle) into the common graph schema.

By default this reads the curated sample bundle shipped in
``data/sample/mitre_attack_sample.json`` so the whole system is runnable
offline and deterministically for tests/demos. ``fetch_live_bundle`` can pull
the real, full Enterprise ATT&CK STIX bundle from the official MITRE CTI
GitHub repository when network access and a larger corpus are desired; the
live bundle is parsed by the same ``parse_stix_bundle`` used for the sample
data (real STIX carries additional fields we simply ignore).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from cti_graphrag.graph.schema import EdgeType, GraphEdge, GraphNode, KnowledgeGraphData, NodeType

SAMPLE_BUNDLE_PATH = Path(__file__).resolve().parents[3] / "data" / "sample" / "mitre_attack_sample.json"

LIVE_BUNDLE_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
    "enterprise-attack/enterprise-attack.json"
)

_RELATIONSHIP_TYPE_MAP = {
    "uses": EdgeType.USES,
    "exploits": EdgeType.EXPLOITS,
    "mitigates": EdgeType.MITIGATES,
    "targets": EdgeType.TARGETS,
    "attributed-to": EdgeType.ATTRIBUTED_TO,
}

_TYPE_MAP = {
    "intrusion-set": NodeType.THREAT_ACTOR,
    "malware": NodeType.MALWARE,
    "attack-pattern": NodeType.TECHNIQUE,
    "x-mitre-tactic": NodeType.TACTIC,
    "course-of-action": NodeType.MITIGATION,
    "campaign": NodeType.CAMPAIGN,
    "identity": NodeType.SECTOR,
}


def load_sample_bundle() -> dict:
    with open(SAMPLE_BUNDLE_PATH, encoding="utf-8") as f:
        return json.load(f)


def fetch_live_bundle(dest: Path | None = None, timeout: float = 30.0) -> dict:
    """Download the full official Enterprise ATT&CK STIX bundle.

    This is a large (tens of MB) file and requires network access; it is not
    used by default. When ``dest`` is given the raw bundle is cached there.
    """
    with urllib.request.urlopen(LIVE_BUNDLE_URL, timeout=timeout) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data))
    return data


def parse_stix_bundle(bundle: dict) -> KnowledgeGraphData:
    """Normalize a (simplified or real) STIX bundle into GraphNode/GraphEdge records."""
    graph = KnowledgeGraphData()
    objects = bundle.get("objects", [])

    nodes_by_id: dict[str, GraphNode] = {}

    for obj in objects:
        stix_type = obj.get("type")
        node_type = _TYPE_MAP.get(stix_type)
        if node_type is None:
            continue

        obj_id = obj["id"]
        name = obj.get("name", obj_id)
        aliases = tuple(obj.get("aliases", []))
        external_id = obj.get("external_id")
        properties = {k: v for k, v in obj.items() if k not in {"type", "id", "name", "aliases"}}
        if external_id:
            properties.setdefault("external_id", external_id)

        node = GraphNode(id=obj_id, type=node_type, name=name, aliases=aliases, properties=properties)
        nodes_by_id[obj_id] = node

        # Technique -> Tactic (BELONGS_TO) and sub-technique parent relationship
        if node_type is NodeType.TECHNIQUE:
            for tactic_ref in obj.get("tactic_refs", []):
                graph.edges.append(GraphEdge(obj_id, tactic_ref, EdgeType.BELONGS_TO))
            parent_ref = obj.get("parent_ref")
            if parent_ref:
                graph.edges.append(GraphEdge(obj_id, parent_ref, EdgeType.SUBTECHNIQUE_OF))

    graph.nodes.extend(nodes_by_id.values())

    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        rel_type = _RELATIONSHIP_TYPE_MAP.get(obj.get("relationship_type"))
        if rel_type is None:
            continue
        graph.edges.append(
            GraphEdge(
                source_id=obj["source_ref"],
                target_id=obj["target_ref"],
                type=rel_type,
            )
        )

    return graph


def load_mitre_attack(use_live: bool = False, cache_path: Path | None = None) -> KnowledgeGraphData:
    """Load MITRE ATT&CK data as normalized graph nodes/edges.

    Args:
        use_live: fetch the full official Enterprise ATT&CK bundle instead of
            the bundled sample. Requires network access.
        cache_path: optional path to cache the live bundle for reuse.
    """
    bundle = fetch_live_bundle(dest=cache_path) if use_live else load_sample_bundle()
    return parse_stix_bundle(bundle)
