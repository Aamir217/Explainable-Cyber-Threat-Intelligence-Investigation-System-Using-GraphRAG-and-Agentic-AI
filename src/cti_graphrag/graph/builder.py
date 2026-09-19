"""Build a populated GraphStore from all ingestion sources."""

from __future__ import annotations

from cti_graphrag.graph.store import GraphStore, InMemoryGraphStore
from cti_graphrag.ingestion.mitre_attack import load_mitre_attack
from cti_graphrag.ingestion.nvd_cve import load_nvd_cves


def build_knowledge_graph(store: GraphStore | None = None, use_live_sources: bool = False) -> GraphStore:
    """Ingest MITRE ATT&CK + NVD/CVE data and load it into a GraphStore.

    Defaults to the bundled sample corpus and an in-memory store so the
    system runs with zero external dependencies. Pass ``use_live_sources``
    to pull the full live MITRE ATT&CK bundle instead (network required).
    """
    store = store or InMemoryGraphStore()

    mitre_data = load_mitre_attack(use_live=use_live_sources)
    cve_data = load_nvd_cves(use_live=False)  # live NVD ingestion needs explicit CVE IDs; see nvd_cve.fetch_live_cves

    # CVE nodes/products must exist before MITRE's "exploits" edges (which
    # reference CVE node ids) are added, since InMemoryGraphStore drops edges
    # whose endpoints are unknown at add-time.
    store.add_nodes(cve_data.nodes)
    store.add_edges(cve_data.edges)
    store.add_nodes(mitre_data.nodes)
    store.add_edges(mitre_data.edges)

    return store
