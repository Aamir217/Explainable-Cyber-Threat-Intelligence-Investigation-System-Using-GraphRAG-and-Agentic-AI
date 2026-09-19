import dataclasses

import pytest

import cti_graphrag.llm as llm_module
from cti_graphrag.llm import OllamaLLM, TemplateLLM, get_llm, is_ollama_available


def test_is_ollama_available_false_when_no_server():
    # No Ollama server is running in the test environment; this must return
    # False rather than raising, so get_llm() can fall back safely.
    assert is_ollama_available(host="http://localhost:1", timeout=0.5) is False


def test_ollama_llm_raises_connection_error_when_unreachable():
    llm = OllamaLLM(model="llama3.2", host="http://localhost:1", timeout=0.5)
    with pytest.raises(ConnectionError):
        llm.generate("system prompt", "user prompt")


def test_get_llm_falls_back_to_template_when_ollama_unreachable(monkeypatch):
    # Settings is a frozen dataclass; swap the module-level binding llm.py
    # actually reads from rather than mutating it in place.
    patched = dataclasses.replace(llm_module.settings, llm_provider="ollama", ollama_host="http://localhost:1")
    monkeypatch.setattr(llm_module, "settings", patched)

    with pytest.warns(UserWarning):
        result = get_llm()
    assert isinstance(result, TemplateLLM)


def test_get_llm_returns_template_explicitly(monkeypatch):
    patched = dataclasses.replace(llm_module.settings, llm_provider="template")
    monkeypatch.setattr(llm_module, "settings", patched)
    assert isinstance(get_llm(), TemplateLLM)
