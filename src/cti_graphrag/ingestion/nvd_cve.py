"""Ingest CVE/NVD vulnerability data into the common graph schema.

Node IDs follow the ``vulnerability--<CVE-ID>`` convention so that
relationship records emitted by other sources (e.g. MITRE ATT&CK malware
"exploits" edges in the sample bundle) can reference CVEs consistently
without needing to ingest them in a particular order.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from cti_graphrag.graph.schema import EdgeType, GraphEdge, GraphNode, KnowledgeGraphData, NodeType

SAMPLE_CVE_PATH = Path(__file__).resolve().parents[3] / "data" / "sample" / "nvd_cve_sample.json"

# NVD REST API v2.0 -- used only when the caller explicitly opts into a live fetch.
NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def cve_node_id(cve_id: str) -> str:
    return f"vulnerability--{cve_id}"


def product_node_id(product_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", product_name.lower()).strip("-")
    return f"product--{slug}"


def load_sample_cves() -> dict:
    with open(SAMPLE_CVE_PATH, encoding="utf-8") as f:
        return json.load(f)


def fetch_live_cves(cve_ids: list[str], timeout: float = 30.0) -> dict:
    """Fetch specific CVE records live from the NVD REST API.

    NVD identifies CVEs individually via ``cveId`` query params; this issues
    one request per ID (NVD's public API has no bulk-by-id endpoint) and
    normalizes the response into the same shape as the sample fixture.
    """
    cves = []
    for cve_id in cve_ids:
        url = f"{NVD_API_URL}?cveId={cve_id}"
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        for item in payload.get("vulnerabilities", []):
            cve = item.get("cve", {})
            descriptions = cve.get("descriptions", [])
            description = next((d["value"] for d in descriptions if d.get("lang") == "en"), "")
            metrics = cve.get("metrics", {})
            score, severity = None, None
            for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                if key in metrics and metrics[key]:
                    cvss_data = metrics[key][0]["cvssData"]
                    score = cvss_data.get("baseScore")
                    severity = cvss_data.get("baseSeverity", metrics[key][0].get("baseSeverity"))
                    break
            products = []
            for config in cve.get("configurations", []):
                for node in config.get("nodes", []):
                    for match in node.get("cpeMatch", []):
                        products.append(match.get("criteria", ""))
            cves.append(
                {
                    "id": cve.get("id", cve_id),
                    "description": description,
                    "cvss_v3_score": score,
                    "cvss_v3_severity": severity,
                    "published_date": cve.get("published", "")[:10],
                    "affected_products": products,
                }
            )
    return {"cves": cves}


def parse_cve_records(data: dict) -> KnowledgeGraphData:
    graph = KnowledgeGraphData()
    products_seen: dict[str, GraphNode] = {}

    for cve in data.get("cves", []):
        node = GraphNode(
            id=cve_node_id(cve["id"]),
            type=NodeType.CVE,
            name=cve["id"],
            properties={
                "description": cve.get("description", ""),
                "cvss_v3_score": cve.get("cvss_v3_score"),
                "cvss_v3_severity": cve.get("cvss_v3_severity"),
                "published_date": cve.get("published_date"),
            },
        )
        graph.nodes.append(node)

        for product_name in cve.get("affected_products", []):
            pid = product_node_id(product_name)
            if pid not in products_seen:
                products_seen[pid] = GraphNode(id=pid, type=NodeType.PRODUCT, name=product_name)
            graph.edges.append(GraphEdge(node.id, pid, EdgeType.AFFECTS))

    graph.nodes.extend(products_seen.values())
    return graph


def load_nvd_cves(use_live: bool = False, cve_ids: list[str] | None = None) -> KnowledgeGraphData:
    if use_live:
        if not cve_ids:
            raise ValueError("cve_ids is required when use_live=True")
        data = fetch_live_cves(cve_ids)
    else:
        data = load_sample_cves()
    return parse_cve_records(data)
