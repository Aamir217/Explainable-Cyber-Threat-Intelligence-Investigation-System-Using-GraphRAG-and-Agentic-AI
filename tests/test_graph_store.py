import pytest

from cti_graphrag.graph.builder import build_knowledge_graph
from cti_graphrag.graph.schema import EdgeType


@pytest.fixture(scope="module")
def store():
    return build_knowledge_graph()


def test_nodes_and_edges_loaded(store):
    assert len(store.all_nodes()) > 20
    assert len(store.all_edges()) > 30


def test_find_nodes_by_name_exact_and_alias(store):
    results = store.find_nodes_by_name("APT28")
    assert results and results[0].name == "APT28"

    alias_results = store.find_nodes_by_name("Fancy Bear")
    assert alias_results and alias_results[0].name == "APT28"


def test_two_hop_traverse_actor_to_technique(store):
    apt28 = store.find_nodes_by_name("APT28")[0]
    paths = store.traverse([apt28.id], [[EdgeType.USES], [EdgeType.USES]])
    end_names = {p.end.name for p in paths}
    assert "Spearphishing Attachment" in end_names  # via Zebrocy
    assert "Command and Scripting Interpreter" in end_names  # via X-Agent


def test_multi_hop_actor_malware_cve(store):
    apt28 = store.find_nodes_by_name("APT28")[0]
    paths = store.traverse([apt28.id], [[EdgeType.USES], [EdgeType.EXPLOITS]])
    cve_names = {p.end.name for p in paths}
    assert "CVE-2020-0688" in cve_names


def test_shortest_path(store):
    apt28 = store.find_nodes_by_name("APT28")[0]
    cve = store.find_nodes_by_name("CVE-2020-0688")[0]
    path = store.shortest_path(apt28.id, cve.id)
    assert path is not None
    assert path.start.name == "APT28"
    assert path.end.name == "CVE-2020-0688"


def test_neighbors_in_direction(store):
    technique = store.find_nodes_by_name("Spearphishing Attachment")[0]
    actors_using_it = store.neighbors(technique.id, edge_types=[EdgeType.USES], direction="in")
    actor_names = {n.name for _, n in actors_using_it}
    assert {"APT28", "APT29"}.issubset(actor_names)
