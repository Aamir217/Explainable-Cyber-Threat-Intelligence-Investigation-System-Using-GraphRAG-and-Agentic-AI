import pytest

from cti_graphrag.corpus import build_corpus
from cti_graphrag.llm import TemplateLLM


@pytest.fixture(scope="session")
def corpus():
    return build_corpus()


@pytest.fixture(scope="session")
def llm():
    return TemplateLLM()
