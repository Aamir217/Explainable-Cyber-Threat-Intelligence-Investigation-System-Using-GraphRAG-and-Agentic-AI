import pytest

from cti_graphrag.embeddings import HashingEmbedder, cosine_similarity, get_embedder


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(dim=64)
    vec1 = embedder.embed_one("APT28 uses Zebrocy")
    vec2 = embedder.embed_one("APT28 uses Zebrocy")
    assert (vec1 == vec2).all()
    assert abs(float((vec1**2).sum()) - 1.0) < 1e-6


def test_cosine_similarity_prefers_overlapping_text():
    embedder = HashingEmbedder(dim=128)
    query = embedder.embed_one("Zebrocy exploits Microsoft Exchange")
    matrix = embedder.embed(["Zebrocy exploits Microsoft Exchange vulnerabilities", "unrelated cooking recipe text"])
    sims = cosine_similarity(query, matrix)
    assert sims[0] > sims[1]


def test_get_embedder_falls_back_to_hashing_when_sentence_transformers_unavailable(monkeypatch):
    import cti_graphrag.embeddings as embeddings_module

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated unavailable model / no network")

    monkeypatch.setattr(embeddings_module, "SentenceTransformerEmbedder", _boom)

    with pytest.warns(UserWarning):
        embedder = get_embedder("sentence-transformers")
    assert isinstance(embedder, HashingEmbedder)


def test_get_embedder_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_embedder("not-a-real-backend")
