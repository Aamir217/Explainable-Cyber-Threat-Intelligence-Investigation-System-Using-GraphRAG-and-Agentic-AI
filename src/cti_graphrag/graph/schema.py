"""Common graph schema shared by all ingestion sources.

Every ingestion module (MITRE ATT&CK, NVD/CVE, CTI reports) normalizes its
source-specific format into these plain, source-agnostic dataclasses before
handing them to the graph builder. Keeping the schema small and explicit
makes it easy to reason about what a "node" or "edge" means regardless of
which upstream feed produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeType(str, Enum):
    THREAT_ACTOR = "ThreatActor"
    MALWARE = "Malware"
    TECHNIQUE = "Technique"
    TACTIC = "Tactic"
    MITIGATION = "Mitigation"
    CAMPAIGN = "Campaign"
    CVE = "CVE"
    PRODUCT = "Product"
    SECTOR = "Sector"


class EdgeType(str, Enum):
    USES = "USES"
    BELONGS_TO = "BELONGS_TO"
    SUBTECHNIQUE_OF = "SUBTECHNIQUE_OF"
    EXPLOITS = "EXPLOITS"
    AFFECTS = "AFFECTS"
    MITIGATES = "MITIGATES"
    TARGETS = "TARGETS"
    ATTRIBUTED_TO = "ATTRIBUTED_TO"


@dataclass(frozen=True)
class GraphNode:
    id: str
    type: NodeType
    name: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    properties: dict = field(default_factory=dict)

    def all_names(self) -> list[str]:
        return [self.name, *self.aliases]


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    type: EdgeType
    properties: dict = field(default_factory=dict)


@dataclass
class KnowledgeGraphData:
    """Container of normalized nodes/edges produced by ingestion, ready for GraphBuilder."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def merge(self, other: "KnowledgeGraphData") -> None:
        self.nodes.extend(other.nodes)
        self.edges.extend(other.edges)
