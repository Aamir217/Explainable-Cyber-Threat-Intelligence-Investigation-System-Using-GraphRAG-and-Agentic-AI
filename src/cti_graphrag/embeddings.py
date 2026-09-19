"""Text embedding abstraction.

Two backends are supported:

* ``hashing`` (default) -- a deterministic, dependency-free hashed
  bag-of-words embedding. It has no external model download, runs fully
  offline, and is fast enough for evaluation loops. It behaves like a crude
  lexical-semantic hybrid: exact/overlapping vocabulary scores highly, which
  is a reasonable stand-in for semantic similarity on the technical,
  vocabulary-heavy CTI domain used here.
* ``sentence-transformers`` -- a real sentence embedding model, used when
  installed and explicitly selected (e.g. via ``EMBEDDING_BACKEND`` env var),
  for higher-quality semantic retrieval in a real deployment.

Both implement the same ``embed(texts) -> np.ndarray`` interface so the rest
of the retrieval stack is agnostic to which is active.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray: ...

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """Deterministic hashed bag-of-words embedding (no model download needed)."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _hash_index(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()  # noqa: S324 (not cryptographic use)
        return int(digest, 16) % self.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokenize(text):
                vectors[i, self._hash_index(token)] += 1.0
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


class SentenceTransformerEmbedder(Embedder):
    """Wraps a real, locally-run sentence-transformers model.

    The model is downloaded once from HuggingFace Hub on first use (a few
    tens of MB for the default ``all-MiniLM-L6-v2``) and cached locally
    (``~/.cache/huggingface``); every call after that runs fully offline on
    CPU. This is what gives the system genuine semantic similarity instead
    of the lexical hashing embedder's bag-of-words approximation.
    """

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        from cti_graphrag.config import settings

        self._model = SentenceTransformer(model_name or settings.embedding_model)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True))


def get_embedder(backend: str = "sentence-transformers", **kwargs) -> Embedder:
    """Build the configured embedder, falling back to the offline hashing
    embedder if ``sentence-transformers`` (or its model weights) can't be
    loaded -- e.g. the package isn't installed, or there's no network access
    on first run to fetch the model.
    """
    if backend == "hashing":
        return HashingEmbedder(**kwargs)
    if backend == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder(**kwargs)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any load failure should fall back, not crash
            import warnings

            warnings.warn(
                f"Could not load sentence-transformers embedder ({exc!r}); "
                "falling back to the offline hashing embedder. Install "
                "'sentence-transformers' and ensure network access on first "
                "run to use real local embeddings.",
                stacklevel=2,
            )
            return HashingEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector ``a`` (1D) and matrix ``b`` (N x D)."""
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return b_norm @ a_norm
