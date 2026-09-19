"""Central configuration, read from environment variables with sane offline defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "template")  # "anthropic" | "openai" | "template"
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-5")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    # Embeddings
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "hashing")  # "hashing" | "sentence-transformers"

    # Graph store
    graph_backend: str = os.getenv("GRAPH_BACKEND", "memory")  # "memory" | "neo4j"
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")

    # Retrieval
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    max_graph_hops: int = int(os.getenv("MAX_GRAPH_HOPS", "3"))


settings = Settings()
