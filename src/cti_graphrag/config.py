"""Central configuration, read from environment variables with sane offline defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # LLM -- defaults to a local Ollama server (no API key, no cloud calls).
    # Falls back to the deterministic TemplateLLM automatically if Ollama
    # isn't reachable (see llm.get_llm()), so the system still runs with
    # zero setup; install Ollama + `ollama pull <model>` to use a real model.
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" | "anthropic" | "openai" | "template"
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    # Embeddings -- defaults to a local sentence-transformers model (real
    # semantic similarity, downloaded once from HuggingFace then fully
    # offline). Falls back to the dependency-free hashing embedder
    # automatically if the package isn't installed (see embeddings.get_embedder()).
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")  # "sentence-transformers" | "hashing"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Graph store
    graph_backend: str = os.getenv("GRAPH_BACKEND", "memory")  # "memory" | "neo4j"
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")

    # Retrieval
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    max_graph_hops: int = int(os.getenv("MAX_GRAPH_HOPS", "3"))


settings = Settings()
