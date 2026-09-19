"""LLM abstraction used for final answer generation.

Three interchangeable providers:

* ``TemplateLLM`` (default, zero-dependency, zero-cost) -- deterministically
  synthesizes an answer by extracting and stitching together the graph
  reasoning paths and document evidence passed in the prompt. This keeps the
  whole system runnable and testable without any API key, and makes answers
  fully reproducible for evaluation.
* ``AnthropicLLM`` -- calls the Anthropic Messages API (requires
  ``ANTHROPIC_API_KEY`` and the optional ``anthropic`` package).
* ``OpenAILLM`` -- calls the OpenAI Chat Completions API (requires
  ``OPENAI_API_KEY`` and the optional ``openai`` package).

Selection is driven by ``config.settings.llm_provider``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from cti_graphrag.config import settings


class LLM(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> str: ...


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
    return TemplateLLM()
