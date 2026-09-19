import pytest

from cti_graphrag.corpus import build_corpus
from cti_graphrag.llm import TemplateLLM


@pytest.fixture(scope="session")
def corpus():
    # Pin the fast, dependency-free hashing embedder for the test suite so
    # tests stay deterministic and don't need network access to fetch a
    # sentence-transformers model (the app's own default, used at runtime).
    return build_corpus(embedding_backend="hashing")


@pytest.fixture(scope="session")
def llm():
    return TemplateLLM()
