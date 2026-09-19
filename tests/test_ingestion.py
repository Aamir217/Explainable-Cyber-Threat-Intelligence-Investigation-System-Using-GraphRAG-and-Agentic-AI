from cti_graphrag.graph.schema import NodeType
from cti_graphrag.ingestion.cti_reports import load_cti_reports
from cti_graphrag.ingestion.mitre_attack import load_mitre_attack
from cti_graphrag.ingestion.nvd_cve import load_nvd_cves


def test_load_mitre_attack_sample():
    graph = load_mitre_attack()
    names = {n.name for n in graph.nodes}
    assert "APT28" in names
    assert "APT29" in names
    assert "Zebrocy" in names
    assert len(graph.edges) > 0


def test_load_nvd_sample():
    graph = load_nvd_cves()
    cve_nodes = [n for n in graph.nodes if n.type is NodeType.CVE]
    assert {n.name for n in cve_nodes} == {"CVE-2020-0688", "CVE-2021-26855", "CVE-2020-4006"}
    product_nodes = [n for n in graph.nodes if n.type is NodeType.PRODUCT]
    assert any(n.name == "Microsoft Exchange Server" for n in product_nodes)


def test_load_cti_reports():
    docs = load_cti_reports()
    assert len(docs) == 5
    doc_ids = {d.doc_id for d in docs}
    assert "RPT-001" in doc_ids
    rpt1 = next(d for d in docs if d.doc_id == "RPT-001")
    assert "APT28" in rpt1.related_actors
    assert "CVE-2020-0688" in rpt1.related_cves
    assert len(rpt1.text) > 0
