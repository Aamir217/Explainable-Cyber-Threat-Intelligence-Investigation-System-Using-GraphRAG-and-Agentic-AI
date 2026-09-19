"""LLM abstraction used for final answer generation.

Four interchangeable providers:

* ``OllamaLLM`` (default) -- calls a locally-running Ollama server
  (https://ollama.com) over its REST API. No API key, no cloud calls: install
  Ollama, run ``ollama pull llama3.2`` (or any other model) and ``ollama
  serve``, and the system uses it automatically.
* ``TemplateLLM`` (automatic fallback) -- deterministically synthesizes an
  answer by extracting and stitching together the graph reasoning paths and
  document evidence passed in the prompt. Used automatically when Ollama
  isn't reachable, so the system still runs with zero setup, and used
  explicitly in tests/evaluation for fully reproducible output.
* ``AnthropicLLM`` -- calls the Anthropic Messages API (requires
  ``ANTHROPIC_API_KEY`` and the optional ``anthropic`` package).
* ``OpenAILLM`` -- calls the OpenAI Chat Completions API (requires
  ``OPENAI_API_KEY`` and the optional ``openai`` package).

Selection is driven by ``config.settings.llm_provider``.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from cti_graphrag.config import settings


class LLM(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> str: ...


class OllamaLLM(LLM):
    """Calls a local Ollama server's chat API (no cloud, no API key)."""

    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 120.0):
        self._model = model or settings.llm_model
        self._host = (host or settings.ollama_host).rstrip("/")
        self._timeout = timeout

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self._host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ConnectionError(
                f"Could not reach Ollama at {self._host} ({exc}). Is `ollama serve` running "
                f"and has the model been pulled (`ollama pull {self._model}`)?"
            ) from exc
        return data.get("message", {}).get("content", "")


def is_ollama_available(host: str | None = None, timeout: float = 1.5) -> bool:
    """Quick reachability check used by ``get_llm()`` to decide whether to fall back."""
    host = (host or settings.ollama_host).rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout):  # noqa: S310
            return True
    except Exception:  # noqa: BLE001 - any failure means "not available", not a crash
        return False


class AnthropicLLM(LLM):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key or settings.anthropic_api_key)
        self._model = model or settings.llm_model

    def generate(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAILLM(LLM):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or settings.openai_api_key)
        self._model = model

    def generate(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""


class TemplateLLM(LLM):
    """Deterministic, dependency-free answer synthesis from structured evidence.

    Expects the user prompt to follow the fixed layout produced by
    ``rag.prompting.build_prompt`` (Question / Graph reasoning paths /
    Supporting document evidence sections), which it parses back out with
    simple regexes -- there is no free-form NLU here, only templated,
    fully-grounded extraction, which is exactly what makes it a reliable,
    reproducible baseline for evaluation.
    """

    def generate(self, system: str, user: str) -> str:
        question = _extract_section(user, "Question") or ""
        graph_block = _extract_section(user, "Graph reasoning paths")
        evidence_block = _extract_section(user, "Supporting document evidence")

        graph_lines = _bullet_lines(graph_block)
        evidence_lines = _bullet_lines(evidence_block)

        sentences: list[str] = []

        if graph_lines:
            sentences.append(
                "Based on verified knowledge graph relationships: " + " ".join(f"{line}." for line in graph_lines)
            )
        if evidence_lines:
            cited = []
            for line in evidence_lines[:3]:
                match = re.match(r"\[(?P<cid>[^\]]+)\]\s*(?P<text>.*)", line)
                if match:
                    cited.append(f"{match.group('text').strip()} [{match.group('cid')}]")
                else:
                    cited.append(line)
            sentences.append("Supporting reporting indicates: " + " ".join(cited))

        if not sentences:
            return (
                f"No graph relationships or supporting documents were found for: '{question}'. "
                "Insufficient evidence to answer with confidence."
            )

        return " ".join(sentences)


def _extract_section(text: str, header: str) -> str:
    pattern = rf"{re.escape(header)}:\s*\n(.*?)(?:\n\n|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _bullet_lines(block: str) -> list[str]:
    if not block:
        return []
    lines = []
    for raw_line in block.splitlines():
        line = raw_line.strip().lstrip("-").strip()
        if line and line.lower() != "none.":
            lines.append(line)
    return lines


def get_llm() -> LLM:
    if settings.llm_provider == "anthropic":
        return AnthropicLLM()
    if settings.llm_provider == "openai":
        return OpenAILLM()
    if settings.llm_provider == "template":
        return TemplateLLM()

    # Default: "ollama". Fall back to the deterministic template automatically
    # if no local server is reachable, so the system still runs out of the box.
    if is_ollama_available():
        return OllamaLLM()

    import warnings

    warnings.warn(
        f"LLM_PROVIDER=ollama but no server was reachable at {settings.ollama_host}; "
        "falling back to the deterministic TemplateLLM. Install Ollama, run "
        f"`ollama pull {settings.llm_model}` and `ollama serve` to use a real local model.",
        stacklevel=2,
    )
    return TemplateLLM()
