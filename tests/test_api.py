import pytest
from fastapi.testclient import TestClient

from cti_graphrag.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_list_systems(client):
    systems = client.get("/systems").json()["systems"]
    assert set(systems) == {"naive_rag", "hybrid_rag", "graph_rag", "agentic_graph_rag"}


def test_query_endpoint(client):
    response = client.post("/query", json={"question": "Which malware does APT28 use?", "system": "graph_rag"})
    assert response.status_code == 200
    body = response.json()
    assert body["system"] == "graph_rag"
    assert body["answer"]
    assert body["graph_paths"]


def test_query_endpoint_unknown_system(client):
    response = client.post("/query", json={"question": "test", "system": "does_not_exist"})
    assert response.status_code == 400


def test_compare_endpoint(client):
    response = client.post("/compare", json={"question": "Which malware does APT28 use?"})
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"naive_rag", "hybrid_rag", "graph_rag", "agentic_graph_rag"}


def test_graph_entity_endpoint(client):
    response = client.get("/graph/entity", params={"name": "APT28", "hops": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["entity"]["name"] == "APT28"
    assert len(body["paths"]) > 0


def test_graph_entity_not_found(client):
    response = client.get("/graph/entity", params={"name": "NoSuchEntityXYZ"})
    assert response.status_code == 404
